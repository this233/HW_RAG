import os
import sys
import time
from rag_system import RAGSystem

def build_vector_database(docs_path: str = "./docs/zh-cn/"):
    """
    Build the vector database from documents.
    
    Args:
        docs_path: Path to the documents directory
    """
    print(f"Building vector database from {docs_path}...")
    start_time = time.time()
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Build vector database
    rag_system.build_vector_db(docs_path)
    
    # Save vector database
    rag_system.save_vector_db()
    
    end_time = time.time()
    print(f"Vector database built in {end_time - start_time:.2f} seconds")
    print(f"Total documents: {len(rag_system.documents)}")

if __name__ == "__main__":
    # Get docs path from command line argument if provided
    docs_path = sys.argv[1] if len(sys.argv) > 1 else "./docs/zh-cn/"
    
    # Check if path exists
    if not os.path.exists(docs_path):
        print(f"Error: Path {docs_path} does not exist")
        sys.exit(1)
    
    # Build vector database
    build_vector_database(docs_path) 