#!/usr/bin/env python3
"""
mem0 Model Comparison Test Suite
Tests different models (via Ollama) using the exact prompts that mem0 uses internally.
This helps evaluate model performance for fact extraction and memory update operations.
"""

import sys
import os
import json
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from textwrap import dedent

# Add API path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Import mem0 prompts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT, get_update_memory_messages

@dataclass
class ModelTestConfig:
    """Configuration for testing a specific model"""
    name: str
    provider: str = "ollama"
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2000
    base_url: str = "http://localhost:11434"
    description: str = ""
    
    def __post_init__(self):
        if not self.model:
            self.model = self.name

@dataclass
class TestResult:
    """Individual test result"""
    model_name: str
    test_type: str
    input_text: str
    output: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None
    token_count: Optional[int] = None
    quality_score: Optional[float] = None

@dataclass
class QualityMetrics:
    """Quality assessment for model outputs"""
    fact_accuracy: float = 0.0  # How accurate are extracted facts
    fact_completeness: float = 0.0  # Did it extract all relevant facts
    memory_operations_accuracy: float = 0.0  # Correct ADD/UPDATE/DELETE operations
    json_compliance: float = 0.0  # Proper JSON formatting
    response_consistency: float = 0.0  # Consistency across similar inputs
    
    @property
    def overall_score(self) -> float:
        return (self.fact_accuracy + self.fact_completeness + 
                self.memory_operations_accuracy + self.json_compliance + 
                self.response_consistency) / 5

class Mem0ModelTester:
    """Test different models using mem0's exact prompts"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        
        # Define models to test
        self.model_configs = [
            ModelTestConfig(
                name="llama3.1:latest",
                description="Meta's Llama 3.1 8B - Default mem0 model"
            ),
            ModelTestConfig(
                name="llama3.2:latest",
                description="Meta's Llama 3.2 3B - Smaller, faster variant"
            ),
            ModelTestConfig(
                name="mistral:latest",
                description="Mistral 7B - Strong open model"
            ),
            ModelTestConfig(
                name="phi3:latest",
                description="Microsoft Phi-3 3.8B - Efficient small model"
            ),
            ModelTestConfig(
                name="gemma2:2b",
                description="Google Gemma 2 2B - Ultra-light model"
            ),
            ModelTestConfig(
                name="qwen2.5:3b",
                description="Alibaba Qwen 2.5 3B - Multilingual capable"
            ),
            ModelTestConfig(
                name="deepseek-r1:1.5b",
                description="DeepSeek R1 1.5B - Reasoning focused"
            ),
            ModelTestConfig(
                name="llama3.2-vision:latest",
                description="Llama 3.2 Vision 11B - Multimodal (text-only test)"
            ),
        ]
        
        # Test conversations for fact extraction
        self.test_conversations = [
            {
                "name": "personal_intro",
                "messages": [
                    {"role": "user", "content": "Hi, I'm Sarah Chen and I work as a data scientist at TechCorp. I love hiking and photography."}
                ],
                "expected_facts": ["Name is Sarah Chen", "Works as a data scientist at TechCorp", "Loves hiking", "Loves photography"]
            },
            {
                "name": "preferences",
                "messages": [
                    {"role": "user", "content": "I prefer Python over Java for ML projects. My favorite framework is PyTorch, though I use TensorFlow at work."}
                ],
                "expected_facts": ["Prefers Python over Java for ML projects", "Favorite framework is PyTorch", "Uses TensorFlow at work"]
            },
            {
                "name": "plans_and_goals", 
                "messages": [
                    {"role": "user", "content": "Next month I'm planning a trip to Japan. I want to visit Tokyo and Kyoto during cherry blossom season."}
                ],
                "expected_facts": ["Planning a trip to Japan next month", "Wants to visit Tokyo and Kyoto", "Wants to visit during cherry blossom season"]
            },
            {
                "name": "health_wellness",
                "messages": [
                    {"role": "user", "content": "I'm allergic to peanuts and shellfish. I try to run 5K three times a week and do yoga on weekends."}
                ],
                "expected_facts": ["Allergic to peanuts", "Allergic to shellfish", "Runs 5K three times a week", "Does yoga on weekends"]
            },
            {
                "name": "complex_conversation",
                "messages": [
                    {"role": "user", "content": "Yesterday I had lunch with my manager John at Sushi Palace. We discussed my promotion to Senior Data Scientist."},
                    {"role": "assistant", "content": "Congratulations on discussing your promotion! How did the conversation go?"},
                    {"role": "user", "content": "It went well! He said I'll get a 15% raise and lead the new recommendation system project starting January."}
                ],
                "expected_facts": ["Had lunch with manager John at Sushi Palace", "Discussed promotion to Senior Data Scientist", "Will get a 15% raise", "Will lead the new recommendation system project", "Project starts in January"]
            }
        ]
        
        # Test cases for memory update operations
        self.memory_update_tests = [
            {
                "name": "simple_add",
                "existing_memory": [],
                "new_facts": ["Name is John", "Lives in San Francisco"],
                "expected_operations": ["ADD", "ADD"]
            },
            {
                "name": "update_preference",
                "existing_memory": [
                    {"id": "0", "text": "Likes cheese pizza"},
                    {"id": "1", "text": "Works as software engineer"}
                ],
                "new_facts": ["Loves pepperoni pizza", "Works as software engineer"],
                "expected_operations": ["UPDATE", "NONE"]
            },
            {
                "name": "complex_update",
                "existing_memory": [
                    {"id": "0", "text": "Name is Sarah"},
                    {"id": "1", "text": "Lives in New York"},
                    {"id": "2", "text": "Enjoys reading books"}
                ],
                "new_facts": ["Lives in San Francisco", "Enjoys reading sci-fi books", "Works at Google"],
                "expected_operations": ["UPDATE", "UPDATE", "ADD"]
            }
        ]

    def check_ollama_availability(self) -> Dict[str, bool]:
        """Check which models are available in Ollama"""
        try:
            import requests
            response = requests.get(f"{self.model_configs[0].base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                available_models = {model['name']: True for model in response.json().get('models', [])}
                return available_models
            return {}
        except Exception as e:
            print(f"Error checking Ollama: {e}")
            return {}

    def test_fact_extraction(self, model_config: ModelTestConfig) -> List[TestResult]:
        """Test fact extraction using mem0's FACT_RETRIEVAL_PROMPT"""
        results = []
        
        try:
            from ollama import Client
            client = Client(host=model_config.base_url)
            
            print(f"\n📝 Testing Fact Extraction: {model_config.name}")
            
            for test_case in self.test_conversations:
                print(f"   Testing: {test_case['name']}")
                
                # Build conversation string
                conversation = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}" 
                    for msg in test_case['messages']
                ])
                
                # Create full prompt
                full_prompt = FACT_RETRIEVAL_PROMPT + "\n\n" + conversation
                
                start_time = time.time()
                try:
                    response = client.chat(
                        model=model_config.model,
                        messages=[{"role": "user", "content": full_prompt}],
                        options={
                            "temperature": model_config.temperature,
                            "num_predict": model_config.max_tokens
                        },
                        format="json"
                    )
                    
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Parse response
                    try:
                        output = json.loads(response['message']['content'])
                        extracted_facts = output.get('facts', [])
                        
                        # Calculate quality score
                        quality_score = self._calculate_fact_quality(
                            extracted_facts, 
                            test_case['expected_facts']
                        )
                        
                        result = TestResult(
                            model_name=model_config.name,
                            test_type="fact_extraction",
                            input_text=test_case['name'],
                            output=extracted_facts,
                            duration_ms=duration_ms,
                            success=True,
                            quality_score=quality_score
                        )
                        
                        print(f"      ✅ {duration_ms:.1f}ms - Extracted {len(extracted_facts)} facts (quality: {quality_score:.2f})")
                        
                    except json.JSONDecodeError as e:
                        result = TestResult(
                            model_name=model_config.name,
                            test_type="fact_extraction",
                            input_text=test_case['name'],
                            output=response['message']['content'],
                            duration_ms=duration_ms,
                            success=False,
                            error=f"JSON parse error: {str(e)}"
                        )
                        print(f"      ❌ {duration_ms:.1f}ms - JSON parse error")
                    
                    results.append(result)
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    result = TestResult(
                        model_name=model_config.name,
                        test_type="fact_extraction",
                        input_text=test_case['name'],
                        output=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
                    
        except Exception as e:
            print(f"   ❌ Failed to initialize Ollama client: {e}")
            
        return results

    def test_memory_updates(self, model_config: ModelTestConfig) -> List[TestResult]:
        """Test memory update operations using mem0's update prompt"""
        results = []
        
        try:
            from ollama import Client
            client = Client(host=model_config.base_url)
            
            print(f"\n🔄 Testing Memory Updates: {model_config.name}")
            
            for test_case in self.memory_update_tests:
                print(f"   Testing: {test_case['name']}")
                
                # Create update prompt
                full_prompt = get_update_memory_messages(
                    test_case['existing_memory'],
                    json.dumps({"facts": test_case['new_facts']})
                )
                
                start_time = time.time()
                try:
                    response = client.chat(
                        model=model_config.model,
                        messages=[{"role": "user", "content": full_prompt}],
                        options={
                            "temperature": model_config.temperature,
                            "num_predict": model_config.max_tokens
                        },
                        format="json"
                    )
                    
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Parse response
                    try:
                        output = json.loads(response['message']['content'])
                        memory_operations = output.get('memory', [])
                        
                        # Calculate quality score
                        quality_score = self._calculate_update_quality(
                            memory_operations,
                            test_case['expected_operations']
                        )
                        
                        result = TestResult(
                            model_name=model_config.name,
                            test_type="memory_update",
                            input_text=test_case['name'],
                            output=memory_operations,
                            duration_ms=duration_ms,
                            success=True,
                            quality_score=quality_score
                        )
                        
                        print(f"      ✅ {duration_ms:.1f}ms - {len(memory_operations)} operations (quality: {quality_score:.2f})")
                        
                    except json.JSONDecodeError as e:
                        result = TestResult(
                            model_name=model_config.name,
                            test_type="memory_update",
                            input_text=test_case['name'],
                            output=response['message']['content'],
                            duration_ms=duration_ms,
                            success=False,
                            error=f"JSON parse error: {str(e)}"
                        )
                        print(f"      ❌ {duration_ms:.1f}ms - JSON parse error")
                    
                    results.append(result)
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    result = TestResult(
                        model_name=model_config.name,
                        test_type="memory_update",
                        input_text=test_case['name'],
                        output=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e)
                    )
                    results.append(result)
                    print(f"      ❌ {duration_ms:.1f}ms - Error: {e}")
                    
        except Exception as e:
            print(f"   ❌ Failed to initialize Ollama client: {e}")
            
        return results

    def _calculate_fact_quality(self, extracted_facts: List[str], expected_facts: List[str]) -> float:
        """Calculate quality score for fact extraction"""
        if not expected_facts:
            return 1.0 if not extracted_facts else 0.5
        
        # Simple scoring: check how many expected facts were captured
        score = 0.0
        for expected in expected_facts:
            # Check if the essence of the expected fact is in any extracted fact
            expected_lower = expected.lower()
            key_words = [w for w in expected_lower.split() if len(w) > 3]
            
            for extracted in extracted_facts:
                extracted_lower = extracted.lower()
                matches = sum(1 for word in key_words if word in extracted_lower)
                if matches >= len(key_words) * 0.6:  # 60% word match
                    score += 1.0
                    break
        
        return score / len(expected_facts)

    def _calculate_update_quality(self, operations: List[Dict], expected_ops: List[str]) -> float:
        """Calculate quality score for memory update operations"""
        if not operations:
            return 0.0
        
        score = 0.0
        for i, op in enumerate(operations):
            if i < len(expected_ops):
                event = op.get('event', 'NONE')
                if event == expected_ops[i]:
                    score += 1.0
                elif event in ['ADD', 'UPDATE', 'DELETE', 'NONE']:
                    score += 0.5  # Partial credit for valid operation
        
        return score / max(len(expected_ops), len(operations))

    def run_comprehensive_test(self):
        """Run all tests for all available models"""
        print("🚀 mem0 Model Comparison Test Suite")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check available models
        print("\n🔍 Checking Ollama model availability...")
        available_models = self.check_ollama_availability()
        
        if not available_models:
            print("❌ Ollama is not running or no models are available!")
            print("   Please start Ollama and pull required models:")
            print("   ollama serve")
            print("   ollama pull llama3.1:latest")
            return
        
        # Filter configs to only available models
        test_configs = []
        for config in self.model_configs:
            if any(config.model in available for available in available_models):
                print(f"   ✅ {config.name}: Available")
                test_configs.append(config)
            else:
                print(f"   ❌ {config.name}: Not available")
        
        if not test_configs:
            print("\n❌ No models available for testing!")
            return
        
        # Run tests
        all_results = []
        for config in test_configs:
            print(f"\n{'='*60}")
            print(f"🤖 Testing Model: {config.name}")
            print(f"   {config.description}")
            
            # Test fact extraction
            fact_results = self.test_fact_extraction(config)
            all_results.extend(fact_results)
            
            # Test memory updates
            update_results = self.test_memory_updates(config)
            all_results.extend(update_results)
            
            time.sleep(1)  # Brief pause between models
        
        self.results = all_results
        
        # Generate report
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        """Generate detailed comparison report"""
        if not self.results:
            print("\n❌ No results to analyze!")
            return
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Group results by model and test type
        model_results = {}
        for result in self.results:
            if result.model_name not in model_results:
                model_results[result.model_name] = {
                    'fact_extraction': [],
                    'memory_update': []
                }
            model_results[result.model_name][result.test_type].append(result)
        
        # Performance Summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("-"*50)
        print(f"{'Model':<25} {'Fact Extract':<15} {'Memory Update':<15} {'Quality':<10}")
        print("-"*50)
        
        model_scores = {}
        
        for model, results in model_results.items():
            # Calculate averages
            fact_times = [r.duration_ms for r in results['fact_extraction'] if r.success]
            update_times = [r.duration_ms for r in results['memory_update'] if r.success]
            
            fact_avg = statistics.mean(fact_times) if fact_times else 0
            update_avg = statistics.mean(update_times) if update_times else 0
            
            # Calculate quality scores
            fact_qualities = [r.quality_score for r in results['fact_extraction'] if r.success and r.quality_score]
            update_qualities = [r.quality_score for r in results['memory_update'] if r.success and r.quality_score]
            
            avg_quality = statistics.mean(fact_qualities + update_qualities) if (fact_qualities + update_qualities) else 0
            
            model_scores[model] = {
                'fact_avg': fact_avg,
                'update_avg': update_avg,
                'quality': avg_quality,
                'combined_score': avg_quality * 100 / ((fact_avg + update_avg) / 2) if (fact_avg + update_avg) > 0 else 0
            }
            
            print(f"{model:<25} {fact_avg:>10.1f}ms {update_avg:>13.1f}ms {avg_quality:>9.2f}")
        
        # Rankings
        print("\n🏆 MODEL RANKINGS")
        print("-"*40)
        
        # Speed ranking
        print("\n⚡ Fastest Models (Average Response Time):")
        speed_ranking = sorted(model_scores.items(), key=lambda x: (x[1]['fact_avg'] + x[1]['update_avg']) / 2)
        for i, (model, scores) in enumerate(speed_ranking[:5], 1):
            avg_time = (scores['fact_avg'] + scores['update_avg']) / 2
            print(f"   {i}. {model}: {avg_time:.1f}ms average")
        
        # Quality ranking
        print("\n🎯 Highest Quality Models:")
        quality_ranking = sorted(model_scores.items(), key=lambda x: x[1]['quality'], reverse=True)
        for i, (model, scores) in enumerate(quality_ranking[:5], 1):
            print(f"   {i}. {model}: {scores['quality']:.2f} quality score")
        
        # Best overall (quality/speed ratio)
        print("\n🌟 Best Overall (Quality/Speed Ratio):")
        overall_ranking = sorted(model_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        for i, (model, scores) in enumerate(overall_ranking[:5], 1):
            print(f"   {i}. {model}: {scores['combined_score']:.1f} combined score")
        
        # Detailed analysis
        print("\n📋 DETAILED ANALYSIS")
        print("-"*40)
        
        for model, results in model_results.items():
            print(f"\n🤖 {model}")
            
            # Success rates
            fact_success = sum(1 for r in results['fact_extraction'] if r.success) / len(results['fact_extraction']) * 100
            update_success = sum(1 for r in results['memory_update'] if r.success) / len(results['memory_update']) * 100
            
            print(f"   Success Rates:")
            print(f"      Fact Extraction: {fact_success:.1f}%")
            print(f"      Memory Updates: {update_success:.1f}%")
            
            # Common errors
            errors = [r.error for r in results['fact_extraction'] + results['memory_update'] if r.error]
            if errors:
                print(f"   Common Issues:")
                unique_errors = list(set(errors))[:3]
                for error in unique_errors:
                    print(f"      - {error[:60]}...")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS")
        print("-"*40)
        
        if overall_ranking:
            best_model = overall_ranking[0][0]
            print(f"\n✅ Best Overall Model: {best_model}")
            print("   - Excellent balance of speed and quality")
            print("   - Suitable for production use with mem0")
            
            if len(overall_ranking) > 1:
                print(f"\n🏃 Fastest Alternative: {speed_ranking[0][0]}")
                print("   - Best for high-throughput scenarios")
                print("   - May sacrifice some quality for speed")
            
            if len(quality_ranking) > 0 and quality_ranking[0][0] != best_model:
                print(f"\n🎖️ Highest Quality: {quality_ranking[0][0]}")
                print("   - Best for accuracy-critical applications")
                print("   - May be slower but more reliable")
        
        # Save results
        self.save_detailed_results()
        
        print(f"\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_detailed_results(self):
        """Save results to JSON file"""
        filename = f"mem0_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            results_data.append({
                "model_name": result.model_name,
                "test_type": result.test_type,
                "input": result.input_text,
                "output": result.output,
                "duration_ms": result.duration_ms,
                "success": result.success,
                "error": result.error,
                "quality_score": result.quality_score
            })
        
        output = {
            "test_metadata": {
                "test_date": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "models_tested": list(set(r.model_name for r in self.results))
            },
            "results": results_data
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Run the test suite"""
    try:
        tester = Mem0ModelTester()
        tester.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()