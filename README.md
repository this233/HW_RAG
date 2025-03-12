# RAG System

A Retrieval-Augmented Generation (RAG) system that supports high concurrency and provides OpenAI-compatible API endpoints.

## Features

- Supports at least 16 concurrent requests without blocking
- Uses FAISS for efficient vector search
- Implements reranking for improved relevance
- Provides streaming responses
- OpenAI-compatible API interface
- Automatic knowledge base construction from Markdown files

## Components

- **Embedding Model**: moka-ai/m3e-large
- **Reranker Model**: BAAI/bge-reranker-large
- **Vector Database**: FAISS
- **API Framework**: FastAPI
- **LLM Integration**: DeepSeek via LLM_api.py

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
python rag_system.py
```

The server will start on http://0.0.0.0:8000

### API Endpoints

The system provides an OpenAI-compatible API endpoint:

- **POST /v1/chat/completions**: Chat completions endpoint

Example request:

```json
{
  "model": "rag-system",
  "messages": [
    {
      "role": "user",
      "content": "什么是OpenHarmony?"
    }
  ],
  "stream": true
}
```

### Building the Knowledge Base

The knowledge base is automatically built from Markdown files in the `./docs/zh-cn/` directory on first startup. If you want to rebuild it:

1. Delete the `vector_db` directory
2. Restart the server

## Concurrency

The system supports at least 16 concurrent requests using:

1. FastAPI's asynchronous request handling
2. Asyncio semaphores for concurrency control
3. ThreadPoolExecutor for CPU-bound tasks
4. Non-blocking streaming responses

## Architecture

1. **Document Processing**: Markdown files are processed and split into chunks with metadata
2. **Vector Database**: Document chunks are embedded and stored in FAISS
3. **Query Processing**:
   - User query is embedded
   - Top 20 relevant documents are retrieved from FAISS
   - Retrieved documents are reranked
   - Top 3 documents are used to generate a prompt for the LLM
4. **Response Generation**: The LLM generates a response based on the retrieved context
5. **Streaming**: Responses are streamed back to the client in real-time

## License

MIT 