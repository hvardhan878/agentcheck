#!/usr/bin/env python3
"""Fixed demo of deterministic replay testing for non-deterministic AI agents."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import agentcheck
from agentcheck.deterministic import BehavioralSignature, ConsistencyScore, TestFailure, DeterministicReplayer
from agentcheck.utils import load_trace, get_current_time
import openai


# Use real OpenAI client
openai_client = openai.OpenAI()


def help_agent(user_question: str, trace: agentcheck.Trace = None) -> str:
    """A simple help agent that answers user questions.
    
    Args:
        user_question: The user's question
        trace: Optional trace instance to record LLM calls
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise, accurate answers."},
        {"role": "user", "content": user_question}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7  # Non-deterministic!
        )
        
        answer = response.choices[0].message.content
        
        # Add the LLM call to the trace if provided
        if trace:
            trace.add_llm_call(
                messages=messages,
                response={
                    "content": answer,
                    "usage": response.usage.model_dump() if response.usage else {},
                    "model": "gpt-4o-mini"
                },
                model="gpt-4o-mini"
            )
        
        return answer
        
    except Exception as e:
        if trace:
            trace.add_llm_call(
                messages=messages,
                model="gpt-4o-mini",
                error=e
            )
        return f"Sorry, I encountered an error: {e}"


class FixedDeterministicReplayer(DeterministicReplayer):
    """Fixed version that properly passes trace to agent functions."""
    
    def establish_baseline(self, agent_func, test_inputs, baseline_name="default"):
        """Establish behavioral baseline for an agent function."""
        print(f"🔄 Establishing baseline '{baseline_name}' with {len(test_inputs)} inputs...")
        
        all_traces = []
        
        for input_data in test_inputs:
            for run in range(self.baseline_runs):
                print(f"  📊 Input {test_inputs.index(input_data) + 1}/{len(test_inputs)}, Run {run + 1}/{self.baseline_runs}")
                
                # Create unique trace file for this run
                trace_file = self.baseline_dir / f"{baseline_name}_input_{test_inputs.index(input_data)}_run_{run}.json"
                
                with agentcheck.Trace(output=trace_file) as trace:
                    trace.metadata["baseline_name"] = baseline_name
                    trace.metadata["input_index"] = test_inputs.index(input_data)
                    trace.metadata["run_number"] = run
                    trace.metadata["input_data"] = str(input_data)[:200]
                    
                    try:
                        # Pass the trace to the agent function
                        result = agent_func(input_data, trace=trace)
                        trace.metadata["result"] = str(result)[:200]
                    except Exception as e:
                        print(f"    ⚠️  Error in run {run + 1}: {e}")
                
                # Load the saved trace
                trace_data = load_trace(trace_file)
                all_traces.append(trace_data)
        
        # Create behavioral signature from all traces
        signature = BehavioralSignature(all_traces)
        self.baselines[baseline_name] = signature
        
        # Save baseline metadata
        baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
        baseline_data = {
            "baseline_name": baseline_name,
            "created_at": get_current_time(),
            "test_inputs": [str(inp)[:100] for inp in test_inputs],
            "baseline_runs": self.baseline_runs,
            "consistency_threshold": self.consistency_threshold,
            "signature": signature.signature,
        }
        
        import json
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Baseline '{baseline_name}' established successfully!")
        print(f"   📈 Analyzed {len(all_traces)} traces")
        print(f"   📁 Saved to {baseline_file}")
    
    def test_consistency(self, agent_func, test_inputs, baseline_name="default"):
        """Test agent consistency against established baseline."""
        if baseline_name not in self.baselines:
            # Try to load baseline from disk
            baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
            if baseline_file.exists():
                import json
                with open(baseline_file, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
                # Reconstruct signature
                self.baselines[baseline_name] = BehavioralSignature([])
                self.baselines[baseline_name].signature = baseline_data["signature"]
            else:
                raise ValueError(f"Baseline '{baseline_name}' not found. Run establish_baseline() first.")
        
        print(f"🧪 Testing consistency against baseline '{baseline_name}'...")
        
        baseline_signature = self.baselines[baseline_name]
        failures = []
        
        for i, input_data in enumerate(test_inputs):
            print(f"  🔍 Testing input {i + 1}/{len(test_inputs)}")
            
            # Run current version once
            trace_file = self.baseline_dir / f"test_{baseline_name}_input_{i}.json"
            
            with agentcheck.Trace(output=trace_file) as trace:
                trace.metadata["test_type"] = "consistency_test"
                trace.metadata["baseline_name"] = baseline_name
                trace.metadata["input_data"] = str(input_data)[:200]
                
                try:
                    # Pass the trace to the agent function
                    result = agent_func(input_data, trace=trace)
                    trace.metadata["result"] = str(result)[:200]
                except Exception as e:
                    print(f"    ⚠️  Error in test: {e}")
            
            # Load and analyze the trace
            trace_data = load_trace(trace_file)
            current_signature = BehavioralSignature([trace_data])
            
            # Calculate consistency score
            consistency_score = ConsistencyScore.calculate(baseline_signature, current_signature)
            
            print(f"    📊 Consistency score: {consistency_score:.3f}")
            
            if consistency_score < self.consistency_threshold:
                failure = TestFailure(
                    input_data=input_data,
                    consistency_score=consistency_score,
                    expected_behavior=baseline_signature.signature,
                    actual_behavior=current_signature.signature,
                    threshold=self.consistency_threshold,
                )
                failures.append(failure)
                print(f"    ❌ FAIL: Score {consistency_score:.3f} < threshold {self.consistency_threshold}")
            else:
                print(f"    ✅ PASS: Score {consistency_score:.3f} >= threshold {self.consistency_threshold}")
        
        if failures:
            print(f"\n❌ {len(failures)}/{len(test_inputs)} tests failed")
            
            # Save failure report
            failure_report = {
                "test_timestamp": get_current_time(),
                "baseline_name": baseline_name,
                "threshold": self.consistency_threshold,
                "total_tests": len(test_inputs),
                "failed_tests": len(failures),
                "failures": [f.to_dict() for f in failures],
            }
            
            import json
            report_file = self.baseline_dir / f"failure_report_{baseline_name}_{get_current_time()[:19].replace(':', '-')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(failure_report, f, indent=2, ensure_ascii=False)
            print(f"📄 Failure report saved to {report_file}")
        else:
            print(f"\n✅ All {len(test_inputs)} tests passed!")
        
        return failures


def main():
    """Demo the fixed deterministic replay testing feature."""
    
    print("🎯 AgentCheck Deterministic Replay Testing Demo (FIXED)")
    print("🔗 Using real OpenAI API with non-deterministic responses")
    print("=" * 60)
    
    # Test inputs for our agent
    test_inputs = [
        "What is Python?",
        "How do I install packages?", 
        "Explain variables in programming",
    ]
    
    print(f"\n📋 Test inputs: {test_inputs}")
    
    # Create fixed replayer
    replayer = FixedDeterministicReplayer(
        consistency_threshold=0.8,
        baseline_runs=3,
        baseline_dir=Path("baselines")
    )
    
    print(f"\n🔄 Step 1: Establishing behavioral baseline...")
    print(f"   This will run each input {replayer.baseline_runs} times to learn the agent's behavior patterns")
    print(f"   💡 Notice: Each run may produce different text, but behavioral patterns should be similar")
    
    # Establish baseline behavior
    replayer.establish_baseline(
        agent_func=help_agent,
        test_inputs=test_inputs,
        baseline_name="help_agent_fixed"
    )
    
    print(f"\n🧪 Step 2: Testing current agent against baseline...")
    print(f"   This will run each input once and check for behavioral consistency")
    print(f"   💡 Even with different text outputs, behavioral patterns should match")
    
    # Test current agent against baseline
    failures = replayer.test_consistency(
        agent_func=help_agent,
        test_inputs=test_inputs,
        baseline_name="help_agent_fixed"
    )
    
    print(f"\n📊 Results Summary:")
    print(f"   Total tests: {len(test_inputs)}")
    print(f"   Failed tests: {len(failures)}")
    print(f"   Success rate: {((len(test_inputs) - len(failures)) / len(test_inputs) * 100):.1f}%")
    
    if failures:
        print(f"\n❌ Test Failures:")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. Input: {failure.input_data}")
            print(f"      Consistency Score: {failure.consistency_score:.3f}")
            print(f"      Threshold: {failure.threshold}")
            print(f"      Reason: Behavioral pattern changed significantly")
    else:
        print(f"\n✅ All tests passed! Agent behavior is consistent.")
    
    print(f"\n📁 Files created:")
    baseline_dir = Path("baselines")
    if baseline_dir.exists():
        for file in sorted(baseline_dir.glob("help_agent_fixed*")):
            print(f"   📄 {file}")
    
    print(f"\n🎉 Demo completed!")


if __name__ == "__main__":
    main() 