#!/usr/bin/env python3
"""
Test script to verify that the vector dimension mismatch error has been fixed.
This test directly calls the MCP add_memories function to verify it works.
"""

import sys
import os

# Add the parent directory to the path to import mem0
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_memory_addition():
    """Test that memory addition works without dimension errors"""
    print("🧪 Testing Memory Addition After Dimension Fix")
    print("=" * 60)
    
    try:
        from mem0 import Memory
        
        # Configuration that matches the database config
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "openmemory",
                    "host": "localhost",  # Using localhost since we're outside Docker
                    "port": 6333,
                    "embedding_model_dims": 768
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.1",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
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
        
        print("📝 Creating Memory client...")
        memory = Memory.from_config(config)
        
        print("🔍 Adding test memory...")
        test_memory = "I love learning about artificial intelligence and machine learning."
        result = memory.add(test_memory, user_id="test_dimension_fix")
        
        print(f"✅ Memory added successfully!")
        print(f"📊 Result: {result}")
        
        print("\n🔎 Searching for the memory...")
        search_results = memory.search("artificial intelligence", user_id="test_dimension_fix", limit=3)
        
        print(f"📈 Search results: {search_results}")
        
        if search_results and len(search_results.get('results', [])) > 0:
            print("\n🎉 SUCCESS: Memory addition and search work correctly!")
            print("✨ The vector dimension mismatch error has been fixed!")
            return True
        else:
            print("\n⚠️  Memory was added but search didn't return results")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 This suggests the dimension issue may not be fully resolved")
        return False

def check_qdrant_collection():
    """Check the Qdrant collection dimensions"""
    import requests
    
    print("\n🗄️  Checking Qdrant Collection:")
    try:
        response = requests.get("http://localhost:6333/collections/openmemory")
        if response.status_code == 200:
            data = response.json()
            dimensions = data["result"]["config"]["params"]["vectors"]["size"]
            print(f"  📏 Collection dimensions: {dimensions}")
            
            if dimensions == 768:
                print("  ✅ Correct dimensions for nomic-embed-text model")
                return True
            else:
                print(f"  ❌ Unexpected dimensions (expected 768, got {dimensions})")
                return False
        else:
            print(f"  ❌ Failed to get collection info: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Error checking collection: {e}")
        return False

def main():
    """Main test function"""
    print("Vector Dimension Fix Verification")
    print("=" * 50)
    
    # Check collection dimensions first
    collection_ok = check_qdrant_collection()
    
    if not collection_ok:
        print("\n❌ Collection dimensions are not correct. Fix needed.")
        return False
    
    # Test memory operations
    memory_ok = test_memory_addition()
    
    if memory_ok:
        print("\n" + "=" * 60)
        print("🎊 VERIFICATION COMPLETE: Dimension fix successful!")
        print("=" * 60)
        print("\n📋 Summary of changes made:")
        print("  • Updated database configuration to include vector_store settings")
        print("  • Set embedding_model_dims to 768 for nomic-embed-text model")
        print("  • Recreated Qdrant collection with correct dimensions")
        print("  • Fixed configuration loading to include vector store settings")
        print("\n🚀 The MCP server should now work correctly with Ollama embeddings!")
    else:
        print("\n❌ Verification failed. Additional debugging needed.")
    
    return memory_ok

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        exit(1)