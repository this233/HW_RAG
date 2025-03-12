import os
import sys
import json
from rag_system import RAGSystem

def test_rag_system():
    """Test the RAG system functionality."""
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    # Try to load existing vector database
    try:
        rag.load_vector_db()
        print(f"Vector database loaded with {len(rag.documents)} documents")
    except FileNotFoundError:
        print("Vector database not found, building new one...")
        rag.build_vector_db()
        rag.save_vector_db()
        print(f"Vector database built with {len(rag.documents)} documents")
    
    # Test queries
    test_queries = [
        "什么是OpenHarmony?",
        "OpenHarmony的架构是什么?",
        "如何开发OpenHarmony应用?"
    ]
    
    for query in test_queries:
        print(f"\n\nTesting query: {query}")
        
        # Test retrieval
        print("\nRetrieval results:")
        retrieval_results = rag.retrieve(query)
        for i, (doc, score, source) in enumerate(retrieval_results[:3]):
            print(f"Result {i+1}:")
            print(f"  Score: {score:.4f}")
            print(f"  Source: {source}")
            print(f"  Path: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content: {doc.page_content[:100]}...")
        
        # Test reranking
        print("\nReranking results:")
        rerank_results = rag.rerank(query, retrieval_results)
        for i, ((doc, orig_score, source), rerank_score) in enumerate(rerank_results):
            print(f"Result {i+1}:")
            print(f"  Original Score: {orig_score:.4f}")
            print(f"  Rerank Score: {rerank_score:.4f}")
            print(f"  Path: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content: {doc.page_content[:100]}...")
        
        # Generate prompt
        print("\nGenerated prompt:")
        prompt = rag._get_rerank_merged(rerank_results, query)
        print(f"{prompt[:500]}...\n")

if __name__ == "__main__":
    test_rag_system() 