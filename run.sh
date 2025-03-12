#!/bin/bash

# Check if vector database exists
if [ ! -d "vector_db" ]; then
    echo "Vector database not found. Building..."
    python build_vector_db.py
fi

# Start the server
echo "Starting RAG system server..."
python rag_system.py 