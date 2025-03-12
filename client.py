import requests
import json
import sys
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional

# API endpoint
API_URL = "http://localhost:8000/v1/chat/completions"

def query_rag_system(query: str, stream: bool = False) -> Dict[str, Any]:
    """
    Send a query to the RAG system.
    
    Args:
        query: The user's query
        stream: Whether to use streaming mode
        
    Returns:
        The response from the RAG system
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "rag-system",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": stream
    }
    
    response = requests.post(API_URL, headers=headers, json=data, stream=stream)
    
    if response.status_code == 200:
        if stream:
            return response
        else:
            return response.json()
    else:
        return {"error": f"Error: {response.status_code} - {response.text}"}

def print_streaming_response(response):
    """
    Print a streaming response from the RAG system.
    
    Args:
        response: The streaming response object
    """
    print("RAG System Response: ", end="", flush=True)
    
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode('utf-8'))
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                        content = chunk["choices"][0]["delta"]["content"]
                        print(content, end="", flush=True)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line.decode('utf-8')}")
    
    print("\n")

async def concurrent_queries(queries: List[str], num_concurrent: int = 16):
    """
    Send multiple queries concurrently to test the system's concurrency handling.
    
    Args:
        queries: List of queries to send
        num_concurrent: Maximum number of concurrent requests
    """
    semaphore = asyncio.Semaphore(num_concurrent)
    start_time = time.time()
    
    async def send_query(query: str, query_id: int):
        async with semaphore:
            print(f"Sending query {query_id}: {query}")
            query_start = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    API_URL,
                    json={
                        "model": "rag-system",
                        "messages": [{"role": "user", "content": query}],
                        "stream": False
                    },
                    headers={"Content-Type": "application/json"}
                ) as response:
                    result = await response.json()
                    query_end = time.time()
                    
                    print(f"Query {query_id} completed in {query_end - query_start:.2f} seconds")
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        print(f"Response for query {query_id}: {content[:100]}...\n")
                    else:
                        print(f"Error in response for query {query_id}: {result}\n")
    
    # Create tasks for all queries
    tasks = [send_query(query, i) for i, query in enumerate(queries)]
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"All {len(queries)} queries completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stream":
            # Streaming mode
            query = input("Enter your query: ")
            response = query_rag_system(query, stream=True)
            print_streaming_response(response)
        elif sys.argv[1] == "--concurrent":
            # Concurrent testing mode
            num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 16
            
            # Generate test queries
            test_queries = [
                "什么是OpenHarmony?",
                "OpenHarmony的架构是什么?",
                "如何开发OpenHarmony应用?",
                "OpenHarmony的设计理念是什么?",
                "OpenHarmony支持哪些设备?",
                "OpenHarmony的安全特性有哪些?",
                "如何贡献代码给OpenHarmony?",
                "OpenHarmony的发展历程是什么?",
                "OpenHarmony与其他操作系统的区别是什么?",
                "OpenHarmony的未来发展方向是什么?"
            ]
            
            # Repeat queries if needed to reach desired number
            while len(test_queries) < num_queries:
                test_queries.extend(test_queries[:num_queries - len(test_queries)])
            
            # Run concurrent queries
            asyncio.run(concurrent_queries(test_queries[:num_queries]))
        else:
            # Normal mode
            query = sys.argv[1]
            response = query_rag_system(query)
            if "choices" in response and len(response["choices"]) > 0:
                print(f"RAG System Response: {response['choices'][0]['message']['content']}")
            else:
                print(f"Error: {response}")
    else:
        print("Usage:")
        print("  python client.py \"Your query here\"")
        print("  python client.py --stream")
        print("  python client.py --concurrent [num_queries]") 