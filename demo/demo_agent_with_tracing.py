"""Demo agent with proper LLM call tracing."""

import os
import sys
from pathlib import Path

# Add agentcheck to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from agentcheck import Trace

SYSTEM_PROMPT = "You are an enthusiastic support agent. Always be helpful and mention the customer's name when responding."


def run_agent_with_tracing(user_msg: str, output_file: str) -> str:
    """Run the demo agent with proper LLM call tracing."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with Trace(output=output_file) as trace:
        # Add metadata
        trace.metadata["function_name"] = "run_agent_with_tracing"
        trace.metadata["user_message"] = user_msg[:100]  # Truncate for storage
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
            )
            
            result = response.choices[0].message.content
            
            # Manually add the LLM call to the trace
            trace.add_llm_call(
                messages=messages,
                response={
                    "content": result,
                    "usage": response.usage.model_dump() if response.usage else {},
                    "model": "gpt-4o-mini"
                },
                model="gpt-4o-mini"
            )
            
            print(f"Agent response: {result}")
            return result
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Add error to trace
            trace.add_llm_call(
                messages=messages,
                model="gpt-4o-mini",
                error=e
            )
            raise


def main() -> None:
    """Main function for the demo agent."""
    scenarios = [
        ("Hi, I'm John Doe. Where is my order #12345? I placed it last week.", "traces/order_inquiry.json"),
        ("I need help setting up my account, my name is Sarah Smith", "traces/account_setup.json"),
        ("Can you help me troubleshoot my billing issue? I'm Mike Johnson", "traces/billing_issue.json"),
        ("I'd like to return an item I purchased yesterday. My name is Emily Chen.", "traces/return_request.json"),
        ("Hi, I'm Alex Rodriguez. Can you help me upgrade my subscription plan?", "traces/upgrade_request.json"),
    ]
    
    print("🚀 Running demo agent with proper tracing...")
    
    for i, (user_message, output_file) in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}/{len(scenarios)}: {output_file}")
        print(f"User message: {user_message}")
        
        try:
            run_agent_with_tracing(user_message, output_file)
            print(f"✅ Trace saved to {output_file}")
        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")
    
    print(f"\n🎉 Generated {len(scenarios)} traces with real LLM calls and costs!")
    print("📁 Check the traces/ directory for the generated files")


if __name__ == "__main__":
    main() 