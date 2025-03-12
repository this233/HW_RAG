# RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, and Hugging Face models. This system provides an OpenAI-compatible API for retrieving and generating answers based on a document corpus.

## Overview

This RAG system processes a collection of Markdown documents, creates vector embeddings, and uses a combination of vector search and reranking to find the most relevant information for user queries. It then uses a Large Language Model (LLM) to generate coherent answers based on the retrieved information.

## Features

- **Document Processing**: Processes Markdown documents with hierarchical structure preservation
- **Vector Search**: Uses FAISS for efficient similarity search
- **Reranking**: Improves retrieval quality with a dedicated reranker model
- **Streaming Responses**: Supports streaming API responses for real-time feedback
- **Concurrency Control**: Manages request queues and concurrent processing
- **OpenAI-Compatible API**: Provides a drop-in replacement for OpenAI's chat completions API
- **GPU Acceleration**: Utilizes GPU for faster processing when available

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG_system.git
   cd RAG_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your documents:
   - Place your Markdown documents in the `docs/zh-cn/` directory
   - The system expects Markdown files with proper heading structure (# for h1, ## for h2, etc.)

4. Build the vector database:
   ```bash
   python build_vector_db.py
   ```

5. Start the server:
   ```bash
   python rag_system.py
   ```
   
   Alternatively, use the provided shell script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

## Usage

### Using the API

The system provides an OpenAI-compatible API endpoint at `/v1/chat/completions`:

```python
import requests
import json

API_URL = "http://localhost:8000/v1/chat/completions"

def query_rag_system(query):
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "rag-system",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()

# Example usage
result = query_rag_system("What is the purpose of this system?")
print(result["choices"][0]["message"]["content"])
```

### Streaming Responses

To use streaming responses:

```python
import requests
import json

API_URL = "http://localhost:8000/v1/chat/completions"

def stream_rag_system(query):
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "rag-system",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": True
    }
    
    response = requests.post(API_URL, headers=headers, json=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                json_str = line_str[6:]
                try:
                    chunk = json.loads(json_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                            content = chunk["choices"][0]["delta"]["content"]
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
    print()

# Example usage
stream_rag_system("What is the purpose of this system?")
```

### Using the Client

The repository includes a client script (`client.py`) that provides a convenient interface to the API:

```bash
python client.py "Your question here"
```

For streaming responses:

```bash
python client.py --stream "Your question here"
```

For concurrent queries testing:

```bash
python client.py --concurrent 10 "Question 1" "Question 2" "Question 3" ...
```

## System Architecture

### Components

1. **DocumentProcessor**: Processes Markdown documents, preserving hierarchical structure
2. **RAGSystem**: Core system that handles embedding, retrieval, reranking, and LLM integration
3. **FastAPI Application**: Provides the API endpoints and handles request routing
4. **RequestQueue**: Manages concurrency and request queuing

### Processing Flow

1. **Document Processing**:
   - Markdown documents are split by headers
   - Each section is further split into semantic chunks
   - Metadata (headers, source) is preserved

2. **Vector Database Creation**:
   - Text chunks are embedded using a Sentence Transformer model
   - Embeddings are stored in a FAISS index

3. **Query Processing**:
   - User query is embedded
   - Vector similarity search retrieves candidate documents
   - Reranker model refines the results
   - Top results are formatted into a prompt
   - LLM generates a response based on the retrieved information

4. **Response Handling**:
   - For streaming requests, chunks are sent as they're generated
   - For non-streaming, the complete response is returned

## Configuration

Key configuration parameters are defined at the top of `rag_system.py`:

- `MAX_CONCURRENT_QUEUE`: Maximum number of concurrent requests (default: 16)
- `MAX_REQUESTS_QUEUE`: Maximum number of requests in queue (default: 64)
- `REQUEST_TIMEOUT`: Timeout for waiting for a request slot (default: 15s)
- `VECTOR_DB_PATH`: Path to store the vector database (default: "vector_db")
- `EMBEDDING_MODEL_NAME`: Path to the embedding model (default: "/mnt/data/m3e-large")
- `RERANKER_MODEL_NAME`: Path to the reranker model (default: "/mnt/data/bge-reranker-large")
- `DOCS_PATH`: Path to the document directory (default: "./docs/zh-cn/")
- `TOP_K_RETRIEVAL`: Number of documents to retrieve (default: 20)
- `TOP_K_RERANK`: Number of documents to keep after reranking (default: 3)

## System Status

You can check the system status at any time by accessing the `/v1/system/status` endpoint:

```bash
curl http://localhost:8000/v1/system/status
```

This will return information about the request queue, concurrent processing, and system configuration.

## Performance Considerations

- The system performs best with a GPU for embedding and reranking
- Adjust `MAX_CONCURRENT_QUEUE` based on your hardware capabilities
- For large document collections, consider increasing `TOP_K_RETRIEVAL` and `TOP_K_RERANK`
- The system is designed to handle concurrent requests efficiently, but performance will depend on your hardware

## License

[Specify your license here]

## Contributing

[Contribution guidelines]

## Acknowledgements

- This project uses models from Hugging Face
- FAISS library for efficient similarity search
- FastAPI for the web framework
- LangChain for document processing utilities 