#!/usr/bin/env python3
"""Test behavioral changes by using a dramatically different agent."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import agentcheck
from agentcheck.deterministic import BehavioralSignature, ConsistencyScore, TestFailure, DeterministicReplayer
from agentcheck.utils import load_trace, get_current_time

# Import from the demo_deterministic module in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from demo_deterministic import FixedDeterministicReplayer

import openai


# Use real OpenAI client
openai_client = openai.OpenAI()


def unhelpful_agent(user_question: str, trace: agentcheck.Trace = None) -> str:
    """A dramatically different agent that should trigger failures.
    
    This agent:
    - Uses a different system prompt
    - Makes multiple LLM calls (vs 1 in baseline)
    - Has different response patterns
    """
    
    # First call - analyze the question
    messages1 = [
        {"role": "system", "content": "You are a question analyzer. Respond with only 'SIMPLE' or 'COMPLEX'."},
        {"role": "user", "content": f"Analyze this question: {user_question}"}
    ]
    
    try:
        response1 = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages1,
            temperature=0.1
        )
        
        analysis = response1.choices[0].message.content
        
        if trace:
            trace.add_llm_call(
                messages=messages1,
                response={
                    "content": analysis,
                    "usage": response1.usage.model_dump() if response1.usage else {},
                    "model": "gpt-4o-mini"
                },
                model="gpt-4o-mini"
            )
        
        # Second call - generate response based on analysis
        if "SIMPLE" in analysis.upper():
            system_prompt = "You are a very brief assistant. Give only one sentence answers."
        else:
            system_prompt = "You are a verbose assistant. Give detailed, multi-paragraph answers."
        
        messages2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        response2 = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages2,
            temperature=0.9  # High temperature for more variation
        )
        
        answer = response2.choices[0].message.content
        
        if trace:
            trace.add_llm_call(
                messages=messages2,
                response={
                    "content": answer,
                    "usage": response2.usage.model_dump() if response2.usage else {},
                    "model": "gpt-4o-mini"
                },
                model="gpt-4o-mini"
            )
        
        return answer
        
    except Exception as e:
        if trace:
            trace.add_llm_call(
                messages=messages1,
                model="gpt-4o-mini",
                error=e
            )
        return f"Error: {e}"


def main():
    """Test a dramatically different agent against existing baseline to demonstrate failure detection."""
    
    print("🔬 Testing Behavioral Changes Detection")
    print("=" * 60)
    
    print("🔄 Changes made to agent:")
    print("   • Multiple LLM calls (2 vs 1 in baseline)")
    print("   • Different system prompts")
    print("   • Higher temperature (0.9 vs 0.7)")
    print("   • Conditional logic based on question analysis")
    print("   • Different response patterns")
    print()
    
    # Test inputs (same as baseline)
    test_inputs = [
        "What is Python?",
        "How do I install packages?", 
        "Explain variables in programming",
    ]
    
    # Create replayer and test against existing baseline
    replayer = FixedDeterministicReplayer(
        consistency_threshold=0.8,
        baseline_runs=3,
        baseline_dir=Path("baselines")
    )
    
    # Load existing baseline
    baseline_name = "help_agent_fixed"
    baseline_file = Path("baselines") / f"{baseline_name}_baseline.json"
    
    if not baseline_file.exists():
        print(f"❌ Baseline '{baseline_name}' not found!")
        print("   Run demo_deterministic.py first to create the baseline.")
        return
    
    print(f"🧪 Testing against baseline '{baseline_name}'...")
    
    failures = replayer.test_consistency(
        agent_func=unhelpful_agent,
        test_inputs=test_inputs,
        baseline_name=baseline_name
    )
    
    print(f"\n📊 Results:")
    print(f"   Total tests: {len(test_inputs)}")
    print(f"   Failed tests: {len(failures)}")
    print(f"   Failure rate: {len(failures)/len(test_inputs)*100:.1f}%")
    
    if failures:
        print(f"\n❌ Test Failures (Expected!):")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. Input: {failure.input_data}")
            print(f"      Consistency Score: {failure.consistency_score:.3f}")
            print(f"      Threshold: {failure.threshold}")
            
            # Show specific differences
            baseline_sig = failure.expected_behavior
            actual_sig = failure.actual_behavior
            
            print(f"      Behavioral Differences:")
            print(f"        Step count - Baseline: {baseline_sig.get('step_count', {}).get('mean', 0):.1f}, Actual: {actual_sig.get('step_count', {}).get('mean', 0):.1f}")
            print(f"        LLM calls - Baseline: {baseline_sig.get('decision_patterns', {}).get('llm_call_count', 0)}, Actual: {actual_sig.get('decision_patterns', {}).get('llm_call_count', 0)}")
            print()
    else:
        print(f"\n⚠️  Unexpected: All tests passed!")
        print(f"    The agent changes might not be dramatic enough to trigger failures.")
    
    print(f"\n💡 This demonstrates:")
    print(f"   • How deterministic testing detects behavioral changes")
    print(f"   • Different step counts and patterns are caught")
    print(f"   • Multiple LLM calls vs single calls are detected")
    print(f"   • Even when responses are still valid, behavior changes are flagged")


if __name__ == "__main__":
    main() 