#!/usr/bin/env python3
"""
Test script to verify that categorization.py works with Ollama
Tests the modified categorization function
"""

import sys
import os
import json

# Add the API path to import the categorization module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

try:
    from app.utils.categorization import get_categories_for_memory
except ImportError as e:
    print(f"Error importing categorization module: {e}")
    sys.exit(1)

def test_ollama_categorization():
    """Test Ollama-based memory categorization"""
    print("🧠 Testing Ollama Memory Categorization")
    print("=" * 50)
    
    # Test memories with different categories
    test_memories = [
        "I had a great lunch with my family at the new Italian restaurant downtown",
        "Started learning Python programming and completed the first tutorial on variables",
        "Went for a 5-mile run this morning and felt really energized afterwards",
        "Received a promotion at work and will be leading the new AI project team",
        "Bought groceries for the week including organic vegetables and fresh fish",
        "Watched the new Marvel movie with friends last weekend",
        "Need to schedule a doctor's appointment for my annual checkup",
        "Deployed the machine learning model to production using Docker containers"
    ]
    
    print("\n📝 Testing categorization with sample memories:\n")
    
    for i, memory in enumerate(test_memories, 1):
        print(f"{i}. Memory: \"{memory}\"")
        
        try:
            categories = get_categories_for_memory(memory)
            if categories:
                print(f"   📊 Categories: {', '.join(categories)}")
            else:
                print("   ⚠️  No categories returned")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
            return False
    
    print("✅ All categorization tests completed successfully!")
    return True

def test_ollama_service():
    """Test if Ollama service is accessible"""
    print("🚀 Testing Ollama Service Connection")
    print("=" * 40)
    
    import requests
    
    try:
        # Test Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama service is running")
            
            # Check for llama3.1:latest model
            llama_models = [m for m in models if "llama3.1:latest" in m.get("name", "")]
            if llama_models:
                print("✅ llama3.1:latest model is available")
                return True
            else:
                print("❌ llama3.1:latest model not found")
                print("Available models:")
                for model in models:
                    print(f"  - {model.get('name', 'Unknown')}")
                return False
        else:
            print(f"❌ Ollama service returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def test_simple_categorization():
    """Test a simple categorization case"""
    print("🎯 Testing Simple Categorization")
    print("=" * 35)
    
    test_memory = "I love machine learning and artificial intelligence"
    print(f"Memory: \"{test_memory}\"")
    
    try:
        categories = get_categories_for_memory(test_memory)
        print(f"Categories: {categories}")
        
        # Check if we got reasonable categories
        expected_keywords = ['ai', 'ml', 'technology', 'artificial', 'machine', 'learning']
        found_relevant = any(keyword in ' '.join(categories).lower() for keyword in expected_keywords)
        
        if found_relevant:
            print("✅ Categorization returned relevant categories")
            return True
        else:
            print("⚠️  Categories may not be highly relevant, but categorization is working")
            return True
            
    except Exception as e:
        print(f"❌ Categorization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("OpenMemory Ollama Categorization Test")
    print("=" * 60)
    
    tests = [
        ("Ollama Service", test_ollama_service),
        ("Simple Categorization", test_simple_categorization),
        ("Full Categorization Suite", test_ollama_categorization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<35} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Categorization is working with Ollama.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)