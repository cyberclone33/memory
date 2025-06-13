#!/usr/bin/env python3
"""
Test models exactly as mem0 uses them internally.
This test creates actual mem0 Memory instances with different model configurations
and measures their performance on real memory operations.
"""

import sys
import os
import json
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from mem0 import Memory
    from mem0.memory.main import Memory as MemoryClass
except ImportError as e:
    print(f"Error importing mem0: {e}")
    sys.exit(1)

@dataclass
class ModelTestResult:
    """Result from testing a model with mem0"""
    model_name: str
    operation: str
    input_text: str
    duration_ms: float
    success: bool
    result: Any = None
    error: Optional[str] = None
    facts_extracted: int = 0
    memory_ids: List[str] = None

class Mem0ActualUsageTester:
    """Test models exactly as mem0 uses them"""
    
    def __init__(self):
        self.results: List[ModelTestResult] = []
        
        # Models to test (based on your available models)
        self.test_models = [
            "llama3.1:latest",
            "llama3:latest", 
            "llama3.2-vision:latest",
            "gemma3:4b",
            "gemma3:1b",
            "phi3:3.8b",
            "qwen3:1.7b",
            "qwen2.5:1.5b",
            "deepseek-r1:1.5b",
            "smollm2:1.7b"
        ]
        
        # Real-world test scenarios that mem0 would encounter
        self.test_scenarios = [
            {
                "name": "user_introduction",
                "texts": [
                    "Hi, I'm Alex Johnson. I work as a software engineer at Google and I love playing tennis on weekends."
                ]
            },
            {
                "name": "preferences_and_habits", 
                "texts": [
                    "I prefer using VSCode for development. My favorite programming language is Python.",
                    "I usually wake up at 6 AM and go for a run. I'm trying to eat healthier, so I've been avoiding sugar."
                ]
            },
            {
                "name": "complex_conversation",
                "texts": [
                    "Yesterday I had a meeting with my manager Sarah about the new AI project.",
                    "She mentioned I might get promoted to Senior Engineer if the project goes well.",
                    "The project involves building a recommendation system using PyTorch."
                ]
            },
            {
                "name": "memory_updates",
                "texts": [
                    "Actually, I've switched from VSCode to Neovim recently.",
                    "I still love Python but I'm also learning Rust now."
                ]
            }
        ]

    def create_mem0_config(self, model_name: str) -> Dict:
        """Create mem0 configuration for a specific model"""
        return {
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
                    "collection_name": f"test_mem0_{model_name.replace(':', '_').replace('.', '_')}",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": 768
                }
            }
        }

    def test_model_with_mem0(self, model_name: str) -> List[ModelTestResult]:
        """Test a model using actual mem0 Memory instance"""
        results = []
        user_id = f"test_user_{model_name.replace(':', '_')}"
        
        print(f"\n🤖 Testing {model_name} with mem0")
        
        try:
            # Create mem0 Memory instance with this model
            config = self.create_mem0_config(model_name)
            memory = Memory.from_config(config)
            
            # Test 1: Adding memories
            print("   📝 Testing memory addition...")
            for scenario in self.test_scenarios[:3]:  # Test first 3 scenarios
                for text in scenario['texts']:
                    start_time = time.time()
                    try:
                        # This is exactly how mem0 is used
                        result = memory.add(text, user_id=user_id)
                        duration_ms = (time.time() - start_time) * 1000
                        
                        # Extract metrics from result
                        facts_count = len(result.get('results', []))
                        memory_ids = [r.get('id') for r in result.get('results', [])]
                        
                        test_result = ModelTestResult(
                            model_name=model_name,
                            operation="memory_add",
                            input_text=f"{scenario['name']}: {text[:50]}...",
                            duration_ms=duration_ms,
                            success=True,
                            result=result,
                            facts_extracted=facts_count,
                            memory_ids=memory_ids
                        )
                        results.append(test_result)
                        
                        print(f"      ✅ {duration_ms:.1f}ms - Extracted {facts_count} facts")
                        
                    except Exception as e:
                        duration_ms = (time.time() - start_time) * 1000
                        test_result = ModelTestResult(
                            model_name=model_name,
                            operation="memory_add",
                            input_text=f"{scenario['name']}: {text[:50]}...",
                            duration_ms=duration_ms,
                            success=False,
                            error=str(e)
                        )
                        results.append(test_result)
                        print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
            
            # Test 2: Searching memories
            print("   🔍 Testing memory search...")
            search_queries = [
                "programming preferences",
                "work and career",
                "daily habits"
            ]
            
            for query in search_queries:
                start_time = time.time()
                try:
                    # Search exactly as mem0 does it
                    search_results = memory.search(query, user_id=user_id, limit=5)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    num_results = len(search_results.get('results', []))
                    
                    test_result = ModelTestResult(
                        model_name=model_name,
                        operation="memory_search",
                        input_text=query,
                        duration_ms=duration_ms,
                        success=True,
                        result=search_results,
                        facts_extracted=num_results
                    )
                    results.append(test_result)
                    
                    print(f"      ✅ {duration_ms:.1f}ms - Found {num_results} memories for '{query}'")
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    test_result = ModelTestResult(
                        model_name=model_name,
                        operation="memory_search",
                        input_text=query,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(test_result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
            
            # Test 3: Memory updates (add conflicting info)
            print("   🔄 Testing memory updates...")
            update_scenario = self.test_scenarios[3]  # memory_updates scenario
            for text in update_scenario['texts']:
                start_time = time.time()
                try:
                    result = memory.add(text, user_id=user_id)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Check if it properly updated existing memories
                    events = [r.get('event') for r in result.get('results', [])]
                    update_count = events.count('UPDATE')
                    add_count = events.count('ADD')
                    
                    test_result = ModelTestResult(
                        model_name=model_name,
                        operation="memory_update",
                        input_text=f"Update: {text[:50]}...",
                        duration_ms=duration_ms,
                        success=True,
                        result=result,
                        facts_extracted=len(result.get('results', []))
                    )
                    results.append(test_result)
                    
                    print(f"      ✅ {duration_ms:.1f}ms - {update_count} updates, {add_count} additions")
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    test_result = ModelTestResult(
                        model_name=model_name,
                        operation="memory_update",
                        input_text=f"Update: {text[:50]}...",
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(test_result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
            
            # Test 4: Get all memories
            print("   📋 Testing get all memories...")
            start_time = time.time()
            try:
                all_memories = memory.get_all(user_id=user_id)
                duration_ms = (time.time() - start_time) * 1000
                
                total_memories = len(all_memories.get('results', []))
                
                test_result = ModelTestResult(
                    model_name=model_name,
                    operation="get_all",
                    input_text="Get all memories",
                    duration_ms=duration_ms,
                    success=True,
                    result=all_memories,
                    facts_extracted=total_memories
                )
                results.append(test_result)
                
                print(f"      ✅ {duration_ms:.1f}ms - Retrieved {total_memories} total memories")
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                test_result = ModelTestResult(
                    model_name=model_name,
                    operation="get_all",
                    input_text="Get all memories",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
                results.append(test_result)
                print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
                
        except Exception as e:
            print(f"   ❌ Failed to create Memory instance: {e}")
            
        return results

    def run_all_tests(self):
        """Run tests for all available models"""
        print("🚀 mem0 Actual Usage Model Test")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check which models are available
        print("\n🔍 Checking model availability...")
        available_models = []
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                installed_models = [m['name'] for m in response.json().get('models', [])]
                for model in self.test_models:
                    if model in installed_models:
                        print(f"   ✅ {model}: Available")
                        available_models.append(model)
                    else:
                        print(f"   ❌ {model}: Not installed")
        except Exception as e:
            print(f"   ❌ Error checking Ollama: {e}")
            return
        
        if not available_models:
            print("\n❌ No models available for testing!")
            return
        
        # Run tests
        all_results = []
        for model in available_models:
            try:
                model_results = self.test_model_with_mem0(model)
                all_results.extend(model_results)
                time.sleep(2)  # Brief pause between models
            except Exception as e:
                print(f"   ❌ Error testing {model}: {e}")
        
        self.results = all_results
        
        # Generate report
        self.generate_performance_report()

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            print("\n❌ No results to analyze!")
            return
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        # Group results by model
        model_stats = {}
        for result in self.results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {
                    'add_times': [],
                    'search_times': [],
                    'update_times': [],
                    'get_all_times': [],
                    'facts_extracted': [],
                    'errors': [],
                    'success_rate': {'total': 0, 'success': 0}
                }
            
            stats = model_stats[result.model_name]
            stats['success_rate']['total'] += 1
            
            if result.success:
                stats['success_rate']['success'] += 1
                if result.operation == 'memory_add':
                    stats['add_times'].append(result.duration_ms)
                    stats['facts_extracted'].append(result.facts_extracted)
                elif result.operation == 'memory_search':
                    stats['search_times'].append(result.duration_ms)
                elif result.operation == 'memory_update':
                    stats['update_times'].append(result.duration_ms)
                elif result.operation == 'get_all':
                    stats['get_all_times'].append(result.duration_ms)
            else:
                stats['errors'].append(result.error)
        
        # Calculate averages and display results
        print("\n📈 MODEL PERFORMANCE METRICS")
        print("-" * 70)
        print(f"{'Model':<20} {'Add (ms)':<12} {'Search (ms)':<12} {'Facts/Add':<12} {'Success':<10}")
        print("-" * 70)
        
        model_scores = {}
        
        for model, stats in model_stats.items():
            add_avg = statistics.mean(stats['add_times']) if stats['add_times'] else 0
            search_avg = statistics.mean(stats['search_times']) if stats['search_times'] else 0
            facts_avg = statistics.mean(stats['facts_extracted']) if stats['facts_extracted'] else 0
            success_rate = (stats['success_rate']['success'] / stats['success_rate']['total'] * 100) if stats['success_rate']['total'] > 0 else 0
            
            # Calculate overall score (lower is better for time, higher for facts/success)
            if add_avg > 0 and search_avg > 0:
                time_score = 1000 / ((add_avg + search_avg) / 2)  # Inverse of average time
                quality_score = facts_avg * (success_rate / 100)
                overall_score = time_score * quality_score
            else:
                overall_score = 0
            
            model_scores[model] = {
                'add_avg': add_avg,
                'search_avg': search_avg,
                'facts_avg': facts_avg,
                'success_rate': success_rate,
                'overall_score': overall_score
            }
            
            print(f"{model:<20} {add_avg:>10.1f} {search_avg:>11.1f} {facts_avg:>11.1f} {success_rate:>8.1f}%")
        
        # Rankings
        print("\n🏆 RANKINGS")
        print("-" * 40)
        
        # Fastest for adding memories
        print("\n⚡ Fastest Memory Addition:")
        add_ranking = sorted([(m, s['add_avg']) for m, s in model_scores.items() if s['add_avg'] > 0], key=lambda x: x[1])
        for i, (model, time) in enumerate(add_ranking[:3], 1):
            print(f"   {i}. {model}: {time:.1f}ms")
        
        # Best fact extraction
        print("\n🎯 Best Fact Extraction:")
        fact_ranking = sorted([(m, s['facts_avg']) for m, s in model_scores.items() if s['facts_avg'] > 0], key=lambda x: x[1], reverse=True)
        for i, (model, facts) in enumerate(fact_ranking[:3], 1):
            print(f"   {i}. {model}: {facts:.1f} facts/input")
        
        # Overall best
        print("\n🌟 Best Overall (Speed + Quality):")
        overall_ranking = sorted([(m, s['overall_score']) for m, s in model_scores.items() if s['overall_score'] > 0], key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(overall_ranking[:3], 1):
            print(f"   {i}. {model}: {score:.1f} score")
        
        # Error analysis
        print("\n⚠️ ERROR ANALYSIS")
        print("-" * 40)
        for model, stats in model_stats.items():
            if stats['errors']:
                print(f"\n{model}:")
                unique_errors = list(set(stats['errors']))[:2]
                for error in unique_errors:
                    print(f"   - {error[:80]}...")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS FOR mem0")
        print("-" * 40)
        
        if overall_ranking:
            best_model = overall_ranking[0][0]
            best_stats = model_scores[best_model]
            
            print(f"\n✅ Best Model: {best_model}")
            print(f"   - Average add time: {best_stats['add_avg']:.1f}ms")
            print(f"   - Average search time: {best_stats['search_avg']:.1f}ms")
            print(f"   - Facts per input: {best_stats['facts_avg']:.1f}")
            print(f"   - Success rate: {best_stats['success_rate']:.1f}%")
            
            # Alternative recommendations
            if len(add_ranking) > 0 and add_ranking[0][0] != best_model:
                print(f"\n🏃 For Speed Priority: {add_ranking[0][0]}")
                print(f"   - Fastest memory operations")
                
            if len(fact_ranking) > 0 and fact_ranking[0][0] != best_model:
                print(f"\n🧠 For Quality Priority: {fact_ranking[0][0]}")
                print(f"   - Best fact extraction accuracy")
        
        # Save results
        self.save_results()
        
        print(f"\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_results(self):
        """Save detailed results to file"""
        filename = f"mem0_actual_usage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = []
        for result in self.results:
            results_data.append({
                "model_name": result.model_name,
                "operation": result.operation,
                "input": result.input_text,
                "duration_ms": result.duration_ms,
                "success": result.success,
                "facts_extracted": result.facts_extracted,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump({
                "test_date": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "results": results_data
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Run the test"""
    try:
        tester = Mem0ActualUsageTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()