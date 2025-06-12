#!/usr/bin/env python3
"""
LLM Performance Comparison Test
Compares speed of different LLMs for categorization and memory saving operations.
Tests both local (Ollama) and cloud (OpenAI) LLM providers.
"""

import sys
import os
import json
import time
import asyncio
import statistics
import importlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add API path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

try:
    from app.utils.memory import get_memory_client
    from app.utils.categorization import get_categories_for_memory
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

@dataclass
class TestResult:
    """Test result data structure"""
    model_name: str
    operation: str
    text: str
    duration_ms: float
    success: bool
    result: Any = None
    error: Optional[str] = None

@dataclass
class LLMConfig:
    """LLM configuration for testing"""
    name: str
    provider: str
    model: str
    config: Dict
    description: str

class LLMPerformanceTester:
    """Performance tester for different LLM configurations"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_memories = [
            "I had a great lunch with my family at the new Italian restaurant downtown",
            "Started learning Python programming and completed the first tutorial on variables", 
            "Went for a 5-mile run this morning and felt really energized afterwards",
            "Received a promotion at work and will be leading the new AI project team",
            "Bought groceries for the week including organic vegetables and fresh fish",
            "Watched the new Marvel movie with friends last weekend",
            "Need to schedule a doctor's appointment for my annual checkup",
            "Deployed the machine learning model to production using Docker containers",
            "Planning a vacation to Japan during cherry blossom season next spring",
            "Completed a successful presentation to the board of directors about Q4 results"
        ]
        
        # Define LLM configurations to test
        self.llm_configs = [
            # All available Ollama models
            LLMConfig(
                name="Ollama-Llama3.1",
                provider="ollama",
                model="llama3.1:latest", 
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Llama 3.1 8B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Llama3",
                provider="ollama",
                model="llama3:latest", 
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Llama 3 8B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Llama3.2-Vision",
                provider="ollama",
                model="llama3.2-vision:latest", 
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Llama 3.2 Vision 11B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Gemma3-4B",
                provider="ollama", 
                model="gemma3:4b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Gemma 3 4B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Gemma3-1B",
                provider="ollama", 
                model="gemma3:1b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Gemma 3 1B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Phi3",
                provider="ollama", 
                model="phi3:3.8b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Phi-3 3.8B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Qwen3",
                provider="ollama", 
                model="qwen3:1.7b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Qwen 3 1.7B via Ollama"
            ),
            LLMConfig(
                name="Ollama-Qwen2.5",
                provider="ollama", 
                model="qwen2.5:1.5b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local Qwen 2.5 1.5B via Ollama"
            ),
            LLMConfig(
                name="Ollama-DeepSeek-R1",
                provider="ollama", 
                model="deepseek-r1:1.5b",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local DeepSeek R1 1.5B via Ollama"
            ),
            LLMConfig(
                name="Ollama-SmolLM2",
                provider="ollama", 
                model="llama3.1:latest",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434"
                },
                description="Local SmolLM2 1.7B via Ollama"
            ),
            # OpenAI models for comparison
            LLMConfig(
                name="OpenAI-GPT4o-mini",
                provider="openai",
                model="gpt-4o-mini",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                description="OpenAI GPT-4o Mini (cloud)"
            ),
            LLMConfig(
                name="OpenAI-GPT3.5-turbo",
                provider="openai", 
                model="gpt-3.5-turbo",
                config={
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                description="OpenAI GPT-3.5 Turbo (cloud)"
            )
        ]

    def check_model_availability(self, llm_config: LLMConfig) -> bool:
        """Check if a model is available for testing"""
        try:
            if llm_config.provider == "ollama":
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [m.get("name", "") for m in models]
                    return any(llm_config.model in model for model in available_models)
                return False
            elif llm_config.provider == "openai":
                # Check if OpenAI API key is available
                return os.getenv("OPENAI_API_KEY") is not None
            return False
        except Exception as e:
            print(f"Error checking {llm_config.name}: {e}")
            return False

    def create_memory_config(self, llm_config: LLMConfig) -> Dict:
        """Create mem0 configuration for the given LLM"""
        base_config = {
            "llm": {
                "provider": llm_config.provider,
                "config": {
                    "model": llm_config.model,
                    **llm_config.config
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"test_{llm_config.name.lower().replace('-', '_')}",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": 768
                }
            }
        }
        
        # Add embedder configuration
        if llm_config.provider == "ollama":
            base_config["embedder"] = {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": "http://localhost:11434"
                }
            }
        else:  # OpenAI
            base_config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        
        return base_config

    def test_categorization_speed(self, llm_config: LLMConfig) -> List[TestResult]:
        """Test categorization speed for a specific LLM"""
        print(f"\n🧠 Testing Categorization: {llm_config.name}")
        print(f"   Description: {llm_config.description}")
        
        results = []
        
        # Backup original config and temporarily override
        original_config_path = os.path.join(os.path.dirname(__file__), 'api', 'mcp_config.json')
        backup_config = None
        
        try:
            # Read original config
            if os.path.exists(original_config_path):
                with open(original_config_path, 'r') as f:
                    backup_config = json.load(f)
            
            # Create test config
            test_config = {"mem0": self.create_memory_config(llm_config)}
            
            # Write test config
            with open(original_config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Force reload of categorization module
            if 'app.utils.categorization' in sys.modules:
                importlib.reload(sys.modules['app.utils.categorization'])
            
            # Test categorization on sample memories
            for i, memory_text in enumerate(self.test_memories[:5]):  # Test first 5 for speed
                print(f"   📝 Testing memory {i+1}/5: '{memory_text[:40]}...'")
                
                start_time = time.time()
                try:
                    categories = get_categories_for_memory(memory_text)
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    result = TestResult(
                        model_name=llm_config.name,
                        operation="categorization",
                        text=memory_text,
                        duration_ms=duration_ms,
                        success=True,
                        result=categories
                    )
                    results.append(result)
                    print(f"      ✅ {duration_ms:.1f}ms - Categories: {categories}")
                    
                except Exception as e:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    result = TestResult(
                        model_name=llm_config.name,
                        operation="categorization", 
                        text=memory_text,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
                    
        finally:
            # Restore original config
            if backup_config:
                with open(original_config_path, 'w') as f:
                    json.dump(backup_config, f, indent=2)
        
        return results

    async def test_memory_operations_speed(self, llm_config: LLMConfig) -> List[TestResult]:
        """Test memory add/search operations speed"""
        print(f"\n💾 Testing Memory Operations: {llm_config.name}")
        
        results = []
        user_id = f"test_user_{llm_config.name.lower().replace('-', '_')}"
        
        try:
            # Create memory client with test config
            from mem0 import Memory
            config = self.create_memory_config(llm_config)
            memory_client = Memory.from_config(config)
            
            # Test memory addition
            for i, memory_text in enumerate(self.test_memories[:3]):  # Test first 3 for speed
                print(f"   📝 Adding memory {i+1}/3: '{memory_text[:40]}...'")
                
                start_time = time.time()
                try:
                    result = memory_client.add(memory_text, user_id=user_id)
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    test_result = TestResult(
                        model_name=llm_config.name,
                        operation="memory_add",
                        text=memory_text,
                        duration_ms=duration_ms,
                        success=True,
                        result=result
                    )
                    results.append(test_result)
                    print(f"      ✅ {duration_ms:.1f}ms - Added successfully")
                    
                except Exception as e:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    test_result = TestResult(
                        model_name=llm_config.name,
                        operation="memory_add",
                        text=memory_text,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(test_result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
            
            # Test memory search
            search_queries = [
                "programming and coding",
                "food and restaurants", 
                "work and career"
            ]
            
            for query in search_queries:
                print(f"   🔍 Searching: '{query}'")
                
                start_time = time.time()
                try:
                    search_results = memory_client.search(query, user_id=user_id, limit=3)
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    test_result = TestResult(
                        model_name=llm_config.name,
                        operation="memory_search",
                        text=query,
                        duration_ms=duration_ms,
                        success=True,
                        result=search_results
                    )
                    results.append(test_result)
                    
                    num_results = len(search_results.get('results', []))
                    print(f"      ✅ {duration_ms:.1f}ms - Found {num_results} results")
                    
                except Exception as e:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    test_result = TestResult(
                        model_name=llm_config.name,
                        operation="memory_search",
                        text=query,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(test_result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
                    
        except Exception as e:
            print(f"   ❌ Failed to create memory client: {e}")
            
        return results

    def run_all_tests(self):
        """Run performance tests for all available LLMs"""
        print("🚀 LLM Performance Comparison Test")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        available_configs = []
        
        # Check which models are available
        print("\n🔍 Checking model availability...")
        for config in self.llm_configs:
            if self.check_model_availability(config):
                print(f"   ✅ {config.name}: Available")
                available_configs.append(config)
            else:
                print(f"   ❌ {config.name}: Not available")
        
        if not available_configs:
            print("\n❌ No LLMs available for testing!")
            print("   Make sure Ollama is running or OpenAI API key is set.")
            return
        
        # Run tests for each available configuration
        all_results = []
        
        for config in available_configs:
            try:
                # Test categorization
                cat_results = self.test_categorization_speed(config)
                all_results.extend(cat_results)
                
                # Test memory operations
                mem_results = asyncio.run(self.test_memory_operations_speed(config))
                all_results.extend(mem_results)
                
            except Exception as e:
                print(f"   ❌ Error testing {config.name}: {e}")
        
        # Store results
        self.results = all_results
        
        # Generate performance report
        self.generate_performance_report()

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            print("\n❌ No results to analyze!")
            return
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        
        # Group results by model and operation
        results_by_model = {}
        for result in self.results:
            if result.model_name not in results_by_model:
                results_by_model[result.model_name] = {}
            if result.operation not in results_by_model[result.model_name]:
                results_by_model[result.model_name][result.operation] = []
            results_by_model[result.model_name][result.operation].append(result)
        
        # Calculate statistics for each model/operation
        print("\n📈 PERFORMANCE STATISTICS")
        print("-" * 50)
        
        performance_summary = {}
        
        for model_name, operations in results_by_model.items():
            print(f"\n🤖 {model_name}")
            performance_summary[model_name] = {}
            
            for operation, results in operations.items():
                successful_results = [r for r in results if r.success]
                
                if successful_results:
                    durations = [r.duration_ms for r in successful_results]
                    avg_time = statistics.mean(durations)
                    median_time = statistics.median(durations)
                    min_time = min(durations)
                    max_time = max(durations)
                    success_rate = len(successful_results) / len(results) * 100
                    
                    performance_summary[model_name][operation] = {
                        'avg_time': avg_time,
                        'median_time': median_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'success_rate': success_rate,
                        'total_tests': len(results)
                    }
                    
                    print(f"   📋 {operation.replace('_', ' ').title()}:")
                    print(f"      Average: {avg_time:.1f}ms")
                    print(f"      Median:  {median_time:.1f}ms")
                    print(f"      Range:   {min_time:.1f}ms - {max_time:.1f}ms")
                    print(f"      Success: {success_rate:.1f}% ({len(successful_results)}/{len(results)})")
                else:
                    print(f"   📋 {operation.replace('_', ' ').title()}: All tests failed")
        
        # Performance rankings
        print("\n🏆 PERFORMANCE RANKINGS")
        print("-" * 30)
        
        operations = ['categorization', 'memory_add', 'memory_search']
        
        for operation in operations:
            print(f"\n⚡ Fastest {operation.replace('_', ' ').title()}:")
            
            # Get models that successfully completed this operation
            model_times = []
            for model_name, ops in performance_summary.items():
                if operation in ops and ops[operation]['success_rate'] > 0:
                    model_times.append((model_name, ops[operation]['avg_time']))
            
            # Sort by average time
            model_times.sort(key=lambda x: x[1])
            
            for i, (model_name, avg_time) in enumerate(model_times, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"   {medal} {i}. {model_name}: {avg_time:.1f}ms")
        
        # Cost analysis (approximate)
        print("\n💰 COST ANALYSIS (Approximate)")
        print("-" * 35)
        
        for model_name in performance_summary.keys():
            config = next(c for c in self.llm_configs if c.name == model_name)
            if config.provider == "ollama":
                print(f"   💚 {model_name}: Free (local inference)")
            elif config.provider == "openai":
                # Rough cost estimates per 1000 operations
                if "gpt-4" in config.model.lower():
                    cost_estimate = "$0.50-1.50"
                else:
                    cost_estimate = "$0.10-0.50"
                print(f"   💸 {model_name}: ~{cost_estimate} per 1000 operations")
        
        # Save detailed results to file
        self.save_results_to_file()
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS")
        print("-" * 20)
        
        # Find fastest overall
        if performance_summary:
            fastest_overall = None
            best_avg = float('inf')
            
            for model_name, ops in performance_summary.items():
                if ops:
                    model_avg = statistics.mean([op['avg_time'] for op in ops.values()])
                    if model_avg < best_avg:
                        best_avg = model_avg
                        fastest_overall = model_name
            
            if fastest_overall:
                print(f"   🚀 Fastest Overall: {fastest_overall} ({best_avg:.1f}ms average)")
            
            # Best local option
            local_models = [name for name, ops in performance_summary.items() 
                           if any(c.provider == "ollama" for c in self.llm_configs if c.name == name)]
            if local_models:
                best_local = min(local_models, key=lambda m: statistics.mean([op['avg_time'] 
                                for op in performance_summary[m].values()]))
                print(f"   🏠 Best Local Option: {best_local} (free, private)")
            
            # Best cloud option
            cloud_models = [name for name, ops in performance_summary.items() 
                           if any(c.provider == "openai" for c in self.llm_configs if c.name == name)]
            if cloud_models:
                best_cloud = min(cloud_models, key=lambda m: statistics.mean([op['avg_time'] 
                                for op in performance_summary[m].values()]))
                print(f"   ☁️  Best Cloud Option: {best_cloud} (pay-per-use)")
        
        print(f"\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_results_to_file(self):
        """Save detailed results to JSON file"""
        filename = f"llm_performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                "model_name": result.model_name,
                "operation": result.operation,
                "text": result.text,
                "duration_ms": result.duration_ms,
                "success": result.success,
                "result": str(result.result) if result.result else None,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                "test_metadata": {
                    "test_date": datetime.now().isoformat(),
                    "total_tests": len(self.results),
                    "test_memories_count": len(self.test_memories)
                },
                "results": results_data
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main test execution"""
    try:
        import importlib
        tester = LLMPerformanceTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()