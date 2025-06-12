#!/usr/bin/env python3
"""
Test script for Ollama embeddings with mem0
Tests local Ollama embedding model functionality
"""

import json
import numpy as np
from typing import List
import sys
import os

# Add the parent directory to the path to import mem0
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mem0 import Memory
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.ollama import OllamaEmbedding
except ImportError as e:
    print(f"Error importing mem0: {e}")
    print("Make sure mem0 is installed: pip install mem0ai")
    sys.exit(1)

def test_direct_ollama_embedding():
    """Test Ollama embeddings directly"""
    print("\n=== Testing Direct Ollama Embedding ===")
    
    try:
        # Create embedding instance with config
        config = BaseEmbedderConfig(
            model="nomic-embed-text",  # Default model
            ollama_base_url="http://localhost:11434"  # Default Ollama URL
        )
        
        embedder = OllamaEmbedding(config)
        
        # Test texts
        test_texts = [
            "The weather is beautiful today",
            "It's a sunny and warm day outside",
            "I love programming in Python",
            "Machine learning is fascinating"
        ]
        
        embeddings = []
        for text in test_texts:
            print(f"\nEmbedding: '{text}'")
            embedding = embedder.embed(text)
            embeddings.append(embedding)
            
            # Show embedding info
            print(f"  - Dimension: {len(embedding)}")
            print(f"  - First 5 values: {embedding[:5]}")
            print(f"  - Norm: {np.linalg.norm(embedding):.4f}")
        
        # Calculate similarities
        print("\n=== Cosine Similarities ===")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                print(f"'{test_texts[i]}' <-> '{test_texts[j]}': {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing embeddings: {e}")
        return False

def test_mem0_with_ollama():
    """Test mem0 with Ollama embeddings"""
    print("\n\n=== Testing Mem0 with Ollama ===")
    
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "ollama_test",
                "path": "./ollama_test_db",
            }
        },
        "llm": {
            "provider": "openai",  # You can also use ollama for LLM
            "config": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434"
            }
        }
    }
    
    try:
        # Initialize memory
        memory = Memory.from_config(config)
        
        # Add some memories
        test_memories = [
            "I love hiking in the mountains during summer",
            "My favorite programming language is Python",
            "I enjoy cooking Italian food on weekends",
            "Mountain biking is my favorite outdoor activity"
        ]
        
        print("\nAdding memories...")
        for mem in test_memories:
            result = memory.add(mem, user_id="test_user")
            print(f"Added: {result}")
        
        # Search memories
        print("\n\nSearching memories...")
        queries = [
            "outdoor activities",
            "programming",
            "hobbies"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = memory.search(query, user_id="test_user", limit=3)
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['memory']} (score: {result.get('score', 'N/A')})")
        
        # Cleanup
        memory.reset()
        
        return True
        
    except Exception as e:
        print(f"Error testing mem0 with Ollama: {e}")
        return False

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def check_ollama_status():
    """Check if Ollama is running"""
    print("=== Checking Ollama Status ===")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running")
            print(f"✓ Available models: {[m['name'] for m in models]}")
            
            # Check for embedding models
            embedding_models = [m['name'] for m in models if 'embed' in m['name'].lower()]
            if embedding_models:
                print(f"✓ Embedding models found: {embedding_models}")
            else:
                print("⚠️  No embedding models found. Install with: ollama pull nomic-embed-text")
            return True
        else:
            print("✗ Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def main():
    """Main test function"""
    print("Ollama Embedding Test Script")
    print("=" * 50)
    
    # Check Ollama status
    if not check_ollama_status():
        print("\n⚠️  Please ensure Ollama is running and has embedding models installed:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull embedding model: ollama pull nomic-embed-text")
        return
    
    # Test direct embedding
    if test_direct_ollama_embedding():
        print("\n✓ Direct Ollama embedding test passed!")
    else:
        print("\n✗ Direct Ollama embedding test failed!")
    
    # Test with mem0 (optional - requires OpenAI API key for LLM)
    if os.getenv("OPENAI_API_KEY"):
        if test_mem0_with_ollama():
            print("\n✓ Mem0 with Ollama test passed!")
        else:
            print("\n✗ Mem0 with Ollama test failed!")
    else:
        print("\n⚠️  Skipping mem0 test (no OPENAI_API_KEY found)")
        print("  To test mem0 integration, set OPENAI_API_KEY environment variable")

if __name__ == "__main__":
    main()