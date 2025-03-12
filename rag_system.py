import os
import json
import time
import asyncio
import faiss
import numpy as np
import torch
import threading
from typing import List, Dict, Any, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from document_processor import DocumentProcessor
import LLM_api

# Constants
MAX_CONCURRENT_QUEUE = 16
MAX_REQUESTS_QUEUE = 64  # Maximum number of requests in queue
REQUEST_TIMEOUT = 15 # Timeout in seconds for waiting for a request slot
VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "/mnt/data/m3e-large"
RERANKER_MODEL_NAME = "/mnt/data/bge-reranker-large"
DOCS_PATH = "./docs/zh-cn/"
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 3

loop = None


class RequestQueue:
    def __init__(self, max_concurrent: int, max_requests: int):
        """
        初始化请求队列，管理并发请求和总请求数量
        
        Args:
            max_concurrent: 最大并发请求数
            max_requests: 最大请求队列长度
        """
        # 使用Semaphore来控制并发数量，更符合asyncio的设计模式
        self.concurrent_semaphore = asyncio.Semaphore(max_concurrent,loop=loop)
        # 使用Semaphore来控制总请求数量
        self.request_semaphore = asyncio.Semaphore(max_requests,loop=loop)
        # 跟踪当前请求数量
        self._request_count = 0
        self._concurrent_count = 0
        # 锁，用于保护计数器的并发访问
        self._lock = asyncio.Lock()
        
    async def acquire(self, timeout: float = REQUEST_TIMEOUT) -> bool:
        """
        尝试获取请求槽位
        
        Args:
            timeout: 获取并发槽位的超时时间（秒），默认使用全局的REQUEST_TIMEOUT
            
        Returns:
            bool: 是否成功获取槽位
        """
        # 1. 首先尝试获取请求队列槽位（非阻塞）
        # 如果请求队列已满，直接拒绝
        if self.is_request_queue_full:
            print("Request queue is full, rejecting request")
            return False
            
        # 2. 尝试获取请求队列槽位
        acquired_request = await self.request_semaphore.acquire()
        if not acquired_request:
            print("Failed to acquire request semaphore")
            return False
        
        async with self._lock:
            self._request_count += 1
            
        try:
            # 3. 尝试在超时时间内获取并发槽位
            try:
                # 使用asyncio的wait_for实现超时机制
                await asyncio.wait_for(self.concurrent_semaphore.acquire(), timeout)
                
                # 4. 成功获取并发槽位，更新计数器
                async with self._lock:
                    self._concurrent_count += 1
                    
                return True
            except asyncio.TimeoutError:
                print(f"Timeout waiting for concurrent slot after {timeout}s")
                async with self._lock:
                    self._request_count -= 1
                self.request_semaphore.release()
                return False
            except Exception as e:
                print(f"Error acquiring concurrent slot: {e}")
                async with self._lock:
                    self._request_count -= 1
                self.request_semaphore.release()
                return False
        except Exception as e:
            print(f"Unexpected error in acquire: {e}")
            return False
                
            
    async def release(self) -> None:
        """释放请求槽位和并发槽位"""
        # 更新计数器
        async with self._lock:
            if self._concurrent_count > 0:
                self._concurrent_count -= 1
            if self._request_count > 0:
                self._request_count -= 1
                
        # 释放信号量
        self.concurrent_semaphore.release()
        self.request_semaphore.release()

    @property
    def request_count(self) -> int:
        """当前请求队列中的请求数量"""
        return self._request_count
        
    @property
    def concurrent_count(self) -> int:
        """当前并发执行的请求数量"""
        return self._concurrent_count
        
    @property
    def is_request_queue_full(self) -> bool:
        """请求队列是否已满"""
        return self.request_semaphore.locked()
        
    @property
    def is_concurrent_queue_full(self) -> bool:
        """并发队列是否已满"""
        return self.concurrent_semaphore.locked()
        
    @property
    def available_request_slots(self) -> int:
        """可用的请求队列槽位数量"""
        return self.request_semaphore._value
        
    @property
    def available_concurrent_slots(self) -> int:
        """可用的并发队列槽位数量"""
        return self.concurrent_semaphore._value

class RAGSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Embedding model initialized: {self.embedding_model}")
        
        # Initialize reranker model
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        print(f"Reranker tokenizer initialized: {self.rerank_tokenizer}")
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(self.device)
        print(f"Reranker model initialized: {self.rerank_model}")
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
        # Initialize request queue for concurrency control
        self.request_queue = RequestQueue(MAX_CONCURRENT_QUEUE, MAX_REQUESTS_QUEUE)
        print(f"Request queue initialized with max concurrent: {MAX_CONCURRENT_QUEUE}, max requests: {MAX_REQUESTS_QUEUE}")
        
        # Initialize thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUEUE)
        print(f"Executor initialized: {self.executor}")
        
        # Add a mutex lock for FAISS GPU operations
        self.faiss_lock = threading.Lock()
        
        print(f"RAGSystem initialized")

    def build_vector_db(self, docs_path: str = DOCS_PATH) -> None:
        """Build the vector database from documents."""
        print(f"Building vector database from {docs_path}...")
        # Process documents
        processor = DocumentProcessor()
        self.documents = processor.process_directory(docs_path)
        
        # Extract text for embedding
        texts = [doc.page_content for doc in self.documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Create FAISS index
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        print(f"Vector database built with {len(self.documents)} documents")
    
    def save_vector_db(self, path: str = VECTOR_DB_PATH) -> None:
        """Save the vector database to disk."""
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Vector database saved to {path}")

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        print(f"FAISS index saved to {os.path.join(path, 'index.faiss')}")
        # Save documents
        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
            # Convert Document objects to dictionaries
            docs_dicts = []
            for doc in self.documents:
                doc_dict = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                docs_dicts.append(doc_dict)
            json.dump(docs_dicts, f, ensure_ascii=False, indent=2)
        
        print(f"Vector database saved to {path}")
    
    def load_vector_db(self, path: str = VECTOR_DB_PATH) -> None:
        """Load the vector database from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector database not found at {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Move index to GPU if available
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load documents
        with open(os.path.join(path, "documents.json"), "r", encoding="utf-8") as f:
            docs_dicts = json.load(f)
            
            # Convert dictionaries back to Document objects
            self.documents = []
            for doc_dict in docs_dicts:
                doc = Document(
                    page_content=doc_dict["page_content"],
                    metadata=doc_dict["metadata"]
                )
                self.documents.append(doc)
        
        print(f"Vector database loaded from {path} with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float, str]]:
        """Retrieve relevant documents using vector similarity."""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Use mutex lock for FAISS operations to ensure thread safety
        with self.faiss_lock:
            # Search in FAISS index
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), 
                top_k
            )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                doc = self.documents[idx]
                distance = distances[0][i]
                # Convert distance to similarity score (1 - normalized distance)
                similarity = 1.0 - distance / 100.0  # Simple normalization
                results.append((doc, similarity, "vector_search"))
        
        return results
    
    def rerank(self, query: str, retrieval_results: List[Tuple[Document, float, str]], top_k: int = TOP_K_RERANK) -> List[Tuple[Tuple[Document, float, str], float]]:
        """Rerank retrieved documents using the reranker model."""
        if not retrieval_results:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, doc.page_content) for doc, _, _ in retrieval_results]
        
        # Tokenize pairs
        features = self.rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # Get reranker scores
        with torch.no_grad():
            scores = self.rerank_model(**features).logits.squeeze(-1).cpu().tolist()
        
        # Combine with original results and sort by reranker score
        reranked_results = [(result, score) for result, score in zip(retrieval_results, scores)]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return reranked_results[:top_k]
    
    def _get_rerank_merged(self, rerank_docs: List, query: str) -> str:
        """构建重排序结果的提示模板"""
        prompt = "基于以下已知信息，简洁和专业的来回答用户的问题。\n"
        prompt += "如果无法从中得到答案，请说 \"无答案\"，不允许在答案中添加编造成分，答案请使用中文。\n\n"
        
        prompt += "-" * 50 + "\n"
        for (doc, orig_score, source), rerank_score in rerank_docs:
            prompt += f"文件路径: {doc.metadata.get('source', '未知')}\n"
            prompt += f"分块位置: " # "h1 > h2 > h3 > h4 > h5 > h6"
            first_h = True
            for i in range(1,7):
                if first_h:
                    prompt += f"h{i} {doc.metadata.get(f'h{i}', '未知')}"
                    first_h = False
                elif doc.metadata.get(f'h{i}'):
                    prompt += f" > h{i} {doc.metadata.get(f'h{i}', '未知')}"
            prompt += f"\n"
            prompt += f"检索来源: {source}\n"
            prompt += f"原始分数: {orig_score:.4f}\n"
            prompt += f"重排序分数: {rerank_score:.4f}\n"
            prompt += f"内容: {doc.page_content}\n"
            prompt += "-" * 50 + "\n"
            
        prompt += f"\n问题: {query}"
        return prompt
    
    async def process_query(self, query: str):
        """Process a query and return a streaming response."""
        acquired = await self.request_queue.acquire()
        
        if not acquired:
            # 检查是请求队列满还是并发队列满导致的拒绝
            if self.request_queue.is_request_queue_full:
                print("服务器请求队列已满, rejecting request")
                # 请求队列已满
                yield JSONResponse(
                    status_code=503,
                    content={"error": "服务器请求队列已满，请稍后再试", "queue_full": True}
                )
            else:
                print(f"等待处理超时（{REQUEST_TIMEOUT}秒）, rejecting request")
                # 并发队列等待超时
                yield JSONResponse(
                    status_code=503,
                    content={"error": f"等待处理超时（{REQUEST_TIMEOUT}秒），请稍后再试", "timeout": True}
                )
            return
       
        
        try:
            # Run retrieval in a thread to avoid blocking
            # loop = asyncio.get_event_loop()
            retrieval_results = await loop.run_in_executor(
                self.executor, 
                self.retrieve,
                query
            )
            
            # Run reranking in a thread
            rerank_results = await loop.run_in_executor(
                self.executor,
                self.rerank,
                query,
                retrieval_results
            )
            
            # Generate prompt for LLM
            prompt = self._get_rerank_merged(rerank_results, query)
            
            # Process with LLM and yield results
            async for chunk in LLM_api.query_deepseek_stream_async(prompt):
                if "error" in chunk:
                    yield json.dumps({"error": chunk["error"]}) + "\n"
                    return
                
                if "final_context" in chunk:
                    continue
                
                if "response" in chunk:
                    # Format response in OpenAI-like format
                    response_chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "rag-system",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk["response"]
                                },
                                "finish_reason": "stop" if chunk.get("done", False) else None
                            }
                        ]
                    }
                    yield json.dumps(response_chunk) + "\n"
        finally:
            # Always release the slot, even if an error occurs
            await self.request_queue.release()

# API Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None

# Initialize FastAPI app
app = FastAPI()

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global loop,rag_system
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    print(f"Event loop: {loop}")
    rag_system = RAGSystem()
    try:
        # Try to load existing vector database
        rag_system.load_vector_db()
    except FileNotFoundError:
        # Build vector database if it doesn't exist
        rag_system.build_vector_db()
        rag_system.save_vector_db()
    
class MyStreamResponse(StreamingResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        except asyncio.exceptions.CancelledError as e:
            print(f"Error in streaming response: {e}")
            raise
        except Exception as e:
            raise

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Extract the last user message as the query
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    query = user_messages[-1].content
    
    if request.stream:
        # Return streaming response
        return MyStreamResponse(
            rag_system.process_query(query),
            media_type="text/event-stream"
        )
    else:
        # For non-streaming, collect all chunks and return as a single response
        chunks = []
        async for chunk in rag_system.process_query(query):
            chunk_data = json.loads(chunk)
            # Check if there's an error
            if "error" in chunk_data:
                raise HTTPException(status_code=503, detail=chunk_data["error"])
            chunks.append(chunk_data)
        
        # Combine all chunks into a single response
        final_content = "".join([chunk["choices"][0]["delta"]["content"] 
                                for chunk in chunks if "choices" in chunk])
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rag-system",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content
                    },
                    "finish_reason": "stop"
                }
            ]
        }

@app.get("/v1/system/status")
async def system_status():
    """Get the current system status including queue information."""
    return {
        "status": "running",
        "request_queue": {
            "current": rag_system.request_queue.request_count,
            "max": MAX_REQUESTS_QUEUE,
            "available": rag_system.request_queue.available_request_slots,
            "is_full": rag_system.request_queue.is_request_queue_full
        },
        "concurrent_queue": {
            "current": rag_system.request_queue.concurrent_count,
            "max": MAX_CONCURRENT_QUEUE,
            "available": rag_system.request_queue.available_concurrent_slots,
            "is_full": rag_system.request_queue.is_concurrent_queue_full
        },
        "timeout": REQUEST_TIMEOUT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 