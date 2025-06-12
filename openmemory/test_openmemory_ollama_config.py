#!/usr/bin/env python3
"""
Test script to verify OpenMemory is correctly configured to use Ollama embeddings
This script tests the configuration and actual embedding functionality
"""

import json
import os
import sys
import requests
from pathlib import Path

# Add the parent directory to the path to import mem0
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ollama_service():
    """Test if Ollama service is running and accessible"""
    print("=== Testing Ollama Service ===")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✓ Ollama service is running")
            print(f"✓ Available models: {len(models)}")
            
            # Check if nomic-embed-text is available
            nomic_available = any(model.get("name") == "nomic-embed-text" for model in models)
            if nomic_available:
                print("✓ nomic-embed-text model is available")
            else:
                print("⚠ nomic-embed-text model not found, will be pulled automatically")
            
            return True
        else:
            print(f"✗ Ollama service returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to Ollama service: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def test_config_files():
    """Test if configuration files are correctly set to use Ollama"""
    print("\n=== Testing Configuration Files ===")
    
    config_files = [
        "/Users/jarvis/Desktop/FUN/memory/openmemory/api/config.json",
        "/Users/jarvis/Desktop/FUN/memory/openmemory/api/default_config.json"
    ]
    
    all_correct = True
    
    for config_path in config_files:
        print(f"\nChecking {Path(config_path).name}:")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            embedder_config = config.get("mem0", {}).get("embedder", {})
            provider = embedder_config.get("provider")
            model = embedder_config.get("config", {}).get("model")
            base_url = embedder_config.get("config", {}).get("ollama_base_url")
            
            if provider == "ollama":
                print("✓ Provider is set to 'ollama'")
            else:
                print(f"✗ Provider is '{provider}', expected 'ollama'")
                all_correct = False
            
            if model == "nomic-embed-text":
                print("✓ Model is set to 'nomic-embed-text'")
            else:
                print(f"✗ Model is '{model}', expected 'nomic-embed-text'")
                all_correct = False
            
            if base_url == "http://localhost:11434":
                print("✓ Base URL is set to 'http://localhost:11434'")
            else:
                print(f"✗ Base URL is '{base_url}', expected 'http://localhost:11434'")
                all_correct = False
                
        except FileNotFoundError:
            print(f"✗ Configuration file not found: {config_path}")
            all_correct = False
        except json.JSONDecodeError:
            print(f"✗ Invalid JSON in configuration file: {config_path}")
            all_correct = False
        except Exception as e:
            print(f"✗ Error reading configuration file: {e}")
            all_correct = False
    
    return all_correct

def test_mem0_ollama_integration():
    """Test mem0 with Ollama configuration"""
    print("\n=== Testing mem0 with Ollama Integration ===")
    
    try:
        from mem0 import Memory
        
        # Load the configuration
        config_path = "/Users/jarvis/Desktop/FUN/memory/openmemory/api/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize Memory with the configuration
        print("Initializing Memory with Ollama configuration...")
        memory = Memory.from_config(config["mem0"])
        
        # Test adding a memory
        test_text = "I love machine learning and artificial intelligence"
        print(f"Adding memory: '{test_text}'")
        
        result = memory.add(test_text, user_id="test_user")
        
        if result:
            print("✓ Successfully added memory using Ollama embeddings")
            print(f"  Memory ID: {result[0].get('id') if result else 'N/A'}")
            
            # Test searching
            print("Testing memory search...")
            search_results = memory.search("artificial intelligence", user_id="test_user")
            
            if search_results:
                print("✓ Successfully searched memories using Ollama embeddings")
                print(f"  Found {len(search_results)} results")
                for i, result in enumerate(search_results[:2]):  # Show first 2 results
                    print(f"  Result {i+1}: {result.get('memory', 'N/A')[:50]}...")
            else:
                print("⚠ Search returned no results")
            
            return True
        else:
            print("✗ Failed to add memory")
            return False
            
    except ImportError as e:
        print(f"✗ Cannot import mem0: {e}")
        print("  Make sure mem0 is installed: pip install mem0ai")
        return False
    except Exception as e:
        print(f"✗ Error testing mem0 with Ollama: {e}")
        return False

def test_direct_ollama_embedding():
    """Test Ollama embeddings directly via HTTP API"""
    print("\n=== Testing Direct Ollama Embedding ===")
    
    test_text = "This is a test for Ollama embeddings"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": test_text
            },
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            print("✓ Successfully generated embedding via Ollama API")
            print(f"✓ Embedding dimensions: {len(embedding)}")
            print(f"✓ Sample values: {embedding[:5]}...")
            return True
        else:
            print(f"✗ Ollama API returned status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error calling Ollama API: {e}")
        return False

def main():
    """Run all tests"""
    print("OpenMemory Ollama Configuration Test")
    print("=" * 50)
    
    tests = [
        ("Ollama Service", test_ollama_service),
        ("Configuration Files", test_config_files),
        ("Direct Ollama Embedding", test_direct_ollama_embedding),
        ("mem0 Ollama Integration", test_mem0_ollama_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! OpenMemory is correctly configured to use Ollama embeddings.")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Please check the configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)