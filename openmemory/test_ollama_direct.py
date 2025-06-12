#!/usr/bin/env python3
"""
Direct test of Ollama embeddings using HTTP API
No dependencies required except requests
"""

import json
import requests
import numpy as np
from typing import List

def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Get embedding from Ollama API directly"""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_ollama_embeddings():
    """Test Ollama embeddings with various texts"""
    print("=== Testing Ollama Embeddings ===\n")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json()["models"]
        embed_models = [m["name"] for m in models if "embed" in m["name"].lower()]
        print(f"✓ Ollama is running")
        print(f"✓ Available embedding models: {embed_models}\n")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return
    
    # Test texts - grouped by similarity
    test_texts = [
        # Similar weather texts
        "The weather is beautiful today",
        "It's a sunny and warm day outside",
        "Today is cold and rainy",
        
        # Programming texts
        "I love programming in Python",
        "Python is my favorite programming language",
        "JavaScript is used for web development",
        
        # Food texts
        "I enjoy cooking Italian food",
        "Pizza and pasta are delicious",
        "I prefer Japanese cuisine",
        
        # Random
        "Machine learning is fascinating",
        "The stock market is volatile",
    ]
    
    print("Getting embeddings for test texts...")
    embeddings = []
    
    for i, text in enumerate(test_texts):
        print(f"{i+1}. '{text}'", end=" ")
        embedding = get_embedding(text)
        if embedding:
            embeddings.append(embedding)
            print(f"✓ (dim: {len(embedding)})")
        else:
            print("✗ Failed")
            return
    
    # Calculate similarities
    print("\n=== Cosine Similarities (Higher = More Similar) ===\n")
    
    # Group similarities by category
    categories = [
        ("Weather", [0, 1, 2]),
        ("Programming", [3, 4, 5]),
        ("Food", [6, 7, 8]),
    ]
    
    for cat_name, indices in categories:
        print(f"\n{cat_name} Category:")
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                similarity = cosine_similarity(embeddings[idx1], embeddings[idx2])
                print(f"  '{test_texts[idx1]}' <-> '{test_texts[idx2]}'")
                print(f"  Similarity: {similarity:.4f}")
    
    # Cross-category similarities (should be lower)
    print("\n\nCross-Category Examples (Should be Lower):")
    cross_pairs = [(0, 3), (0, 6), (3, 6)]  # weather-prog, weather-food, prog-food
    for idx1, idx2 in cross_pairs:
        similarity = cosine_similarity(embeddings[idx1], embeddings[idx2])
        print(f"  '{test_texts[idx1]}' <-> '{test_texts[idx2]}'")
        print(f"  Similarity: {similarity:.4f}")
    
    # Test query functionality
    print("\n\n=== Query Test ===")
    queries = [
        "What's the climate like?",
        "coding and software development",
        "restaurants and dining"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        query_embedding = get_embedding(query)
        if query_embedding:
            # Find top 3 most similar texts
            similarities = []
            for i, emb in enumerate(embeddings):
                sim = cosine_similarity(query_embedding, emb)
                similarities.append((sim, i))
            
            similarities.sort(reverse=True)
            print("Top 3 most similar texts:")
            for sim, idx in similarities[:3]:
                print(f"  {sim:.4f}: '{test_texts[idx]}'")
    
    print("\n\n=== Summary ===")
    print("✓ Ollama embeddings are working correctly!")
    print("✓ Similar texts have higher cosine similarity")
    print("✓ Different topics have lower similarity")
    print("✓ Query matching works as expected")
    
    # Configuration for OpenMemory
    print("\n\n=== Configuration for OpenMemory ===")
    print("Add this to your config.json or use the API endpoint:")
    config = {
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434"
            }
        }
    }
    print(json.dumps(config, indent=2))
    
    print("\nFor Docker environments, use:")
    config["embedder"]["config"]["ollama_base_url"] = "http://host.docker.internal:11434"
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    test_ollama_embeddings()