"""Demo agent showcasing agentcheck functionality."""

import os
import sys
from pathlib import Path

# Add agentcheck to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from agentcheck import trace

SYSTEM_PROMPT = "You are an enthusiastic support agent. Always be helpful and mention the customer's name when responding."


@trace(output="demo_trace.json")
def run_agent(user_msg: str) -> str:
    """Run the demo agent with tracing enabled."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
        print(f"Agent response: {result}")
        return result
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def main() -> None:
    """Main function for the demo agent."""
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
    else:
        user_message = "Hi, I'm John Doe. Where is my order #12345? I placed it last week."
    
    print(f"User message: {user_message}")
    run_agent(user_message)


if __name__ == "__main__":
    main() 