#!/usr/bin/env python3
"""Demo of deterministic replay testing for non-deterministic AI agents."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import agentcheck
import openai


# Use real OpenAI client
openai_client = openai.OpenAI()


@agentcheck.deterministic_replay(
    consistency_threshold=0.8,
    baseline_runs=3,
    baseline_name="help_agent",
    baseline_dir="baselines"
)
def help_agent(user_question: str) -> str:
    """A simple help agent that answers user questions."""
    
    # Real LLM call with tracing
    with agentcheck.Trace() as trace:
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
            
            # Add the LLM call to the trace
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
            return f"Sorry, I encountered an error: {e}"


def main():
    """Demo the deterministic replay testing feature."""
    
    print("🎯 AgentCheck Deterministic Replay Testing Demo")
    print("🔗 Using real OpenAI API with non-deterministic responses")
    print("=" * 60)
    
    # Test inputs for our agent
    test_inputs = [
        "What is Python?",
        "How do I install packages?", 
        "Explain variables in programming",
    ]
    
    print(f"\n📋 Test inputs: {test_inputs}")
    
    # Get the replayer from the decorated function
    replayer = help_agent._deterministic_replayer
    
    print(f"\n🔄 Step 1: Establishing behavioral baseline...")
    print(f"   This will run each input {replayer.baseline_runs} times to learn the agent's behavior patterns")
    print(f"   💡 Notice: Each run may produce different text, but behavioral patterns should be similar")
    
    # Establish baseline behavior
    replayer.establish_baseline(
        agent_func=help_agent,
        test_inputs=test_inputs,
        baseline_name="help_agent"
    )
    
    print(f"\n🧪 Step 2: Testing current agent against baseline...")
    print(f"   This will run each input once and check for behavioral consistency")
    print(f"   💡 Even with different text outputs, behavioral patterns should match")
    
    # Test current agent against baseline
    failures = replayer.test_consistency(
        agent_func=help_agent,
        test_inputs=test_inputs,
        baseline_name="help_agent"
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
        for file in baseline_dir.glob("*"):
            print(f"   📄 {file}")
    
    print(f"\n🎉 Demo completed!")
    print(f"\n💡 Key Benefits:")
    print(f"   • Detects when agent behavior changes unexpectedly")
    print(f"   • Works with non-deterministic LLM outputs")
    print(f"   • Provides detailed failure analysis")
    print(f"   • Integrates seamlessly with existing agentcheck tracing")
    
    print(f"\n🔬 What happened:")
    print(f"   • Each run produced different text responses (due to temperature=0.7)")
    print(f"   • But behavioral patterns (step counts, types, structure) remained consistent")
    print(f"   • This is the power of behavioral testing vs exact output matching!")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Try changing the system prompt and re-run to see failures")
    print(f"   • Integrate into your CI/CD pipeline")
    print(f"   • Set up different baselines for different agent versions")


if __name__ == "__main__":
    main() 