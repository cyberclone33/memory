#!/usr/bin/env python3
"""
Quick model comparison for mem0 usage - tests core operations with minimal scenarios
"""

import sys
import os
import json
import time
from datetime import datetime
import statistics

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mem0 import Memory

class QuickMem0Tester:
    def __init__(self):
        self.models_to_test = [
            "llama3.1:latest",
            "llama3:latest",
            "gemma3:1b",  # Small and fast
            "phi3:3.8b",
            "qwen2.5:1.5b",  # Small model
            "deepseek-r1:1.5b"
        ]
        
        # Simple test case
        self.test_text = "I'm John Smith, a software engineer at Google. I love Python and machine learning."
        self.search_query = "programming preferences"
        
    def test_model(self, model_name: str):
        """Test a single model quickly"""
        print(f"\n🤖 Testing {model_name}")
        
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": model_name,
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
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"quick_test_{model_name.replace(':', '_')}",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": 768
                }
            }
        }
        
        results = {
            'model': model_name,
            'add_time': 0,
            'search_time': 0,
            'facts_count': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Create memory instance
            memory = Memory.from_config(config)
            user_id = f"test_{model_name.replace(':', '_')}"
            
            # Test 1: Add memory
            start = time.time()
            add_result = memory.add(self.test_text, user_id=user_id)
            results['add_time'] = (time.time() - start) * 1000
            results['facts_count'] = len(add_result.get('results', []))
            
            print(f"   ✅ Add: {results['add_time']:.1f}ms - {results['facts_count']} facts")
            
            # Brief pause
            time.sleep(0.5)
            
            # Test 2: Search memory
            start = time.time()
            search_result = memory.search(self.search_query, user_id=user_id, limit=3)
            results['search_time'] = (time.time() - start) * 1000
            
            print(f"   ✅ Search: {results['search_time']:.1f}ms - {len(search_result.get('results', []))} results")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            print(f"   ❌ Error: {e}")
            
        return results

    def run_comparison(self):
        """Run quick comparison of available models"""
        print("🚀 Quick mem0 Model Comparison")
        print("=" * 50)
        
        # Check available models
        available_models = []
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                installed = [m['name'] for m in response.json().get('models', [])]
                for model in self.models_to_test:
                    if model in installed:
                        available_models.append(model)
        except:
            print("❌ Cannot connect to Ollama")
            return
            
        if not available_models:
            print("❌ No test models available")
            return
            
        print(f"\n📋 Testing {len(available_models)} models...")
        
        # Test each model
        all_results = []
        for model in available_models:
            result = self.test_model(model)
            all_results.append(result)
            time.sleep(1)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 RESULTS SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'Add (ms)':<12} {'Search (ms)':<12} {'Facts':<8} {'Total (ms)':<12}")
        print("-"*70)
        
        # Sort by total time
        successful = [r for r in all_results if r['success']]
        successful.sort(key=lambda x: x['add_time'] + x['search_time'])
        
        for r in successful:
            total_time = r['add_time'] + r['search_time']
            print(f"{r['model']:<20} {r['add_time']:>10.1f} {r['search_time']:>11.1f} {r['facts_count']:>7} {total_time:>11.1f}")
        
        # Failed models
        failed = [r for r in all_results if not r['success']]
        if failed:
            print("\n❌ Failed Models:")
            for r in failed:
                print(f"   {r['model']}: {r['error'][:60]}...")
        
        # Best model
        if successful:
            best = successful[0]
            print(f"\n🏆 Fastest Model: {best['model']}")
            print(f"   Total time: {best['add_time'] + best['search_time']:.1f}ms")
            print(f"   Facts extracted: {best['facts_count']}")
            
            # Best quality (most facts)
            best_quality = max(successful, key=lambda x: x['facts_count'])
            if best_quality['model'] != best['model']:
                print(f"\n🧠 Best Quality: {best_quality['model']}")
                print(f"   Facts extracted: {best_quality['facts_count']}")

if __name__ == "__main__":
    tester = QuickMem0Tester()
    tester.run_comparison()