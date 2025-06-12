#!/usr/bin/env python3
"""
Simple test to verify that OpenMemory embedding configuration has been changed to Ollama
This test focuses only on the embedding component, not the full memory system
"""

import json
import requests

def test_embedding_config_change():
    """Test that configuration files correctly specify Ollama for embeddings"""
    print("🔍 Testing OpenMemory Embedding Configuration Change")
    print("=" * 60)
    
    # Test configuration files
    config_files = [
        "/Users/jarvis/Desktop/FUN/memory/openmemory/api/config.json",
        "/Users/jarvis/Desktop/FUN/memory/openmemory/api/default_config.json"
    ]
    
    print("\n📁 Configuration Files:")
    for config_path in config_files:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        embedder = config["mem0"]["embedder"]
        print(f"\n  📄 {config_path.split('/')[-1]}:")
        print(f"    Provider: {embedder['provider']}")
        print(f"    Model: {embedder['config']['model']}")
        print(f"    URL: {embedder['config']['ollama_base_url']}")
        
        # Verify it's changed from OpenAI to Ollama
        assert embedder["provider"] == "ollama", f"Expected 'ollama', got '{embedder['provider']}'"
        assert embedder["config"]["model"] == "nomic-embed-text", f"Expected 'nomic-embed-text', got '{embedder['config']['model']}'"
        assert embedder["config"]["ollama_base_url"] == "http://localhost:11434", f"Expected 'http://localhost:11434', got '{embedder['config']['ollama_base_url']}'"
    
    print("\n✅ Configuration files correctly specify Ollama embeddings")
    
    # Test Ollama service
    print("\n🚀 Ollama Service:")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"  ✅ Ollama service is running ({len(models)} models available)")
        else:
            print(f"  ❌ Ollama service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        return False
    
    # Test embedding functionality
    print("\n🧠 Embedding Test:")
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "Test embedding with Ollama"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            print(f"  ✅ Ollama embedding successful")
            print(f"  📊 Dimensions: {len(embedding)}")
            print(f"  🔢 Sample values: {embedding[:3]}...")
        else:
            print(f"  ❌ Embedding failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS: OpenMemory has been successfully configured to use Ollama embeddings!")
    print("=" * 60)
    print("\n📋 Summary of changes:")
    print("  • Embedder provider: OpenAI → Ollama")
    print("  • Embedder model: text-embedding-3-small → nomic-embed-text")
    print("  • Embedder URL: (OpenAI API) → http://localhost:11434")
    print("\n✨ The embedding model change is complete and working correctly.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_embedding_config_change()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)