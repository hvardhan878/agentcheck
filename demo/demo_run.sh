#!/bin/bash

# demo_run.sh - AgentCheck Demo Script
# Shows the complete workflow: trace → replay → diff → assert

set -e

echo "🤖 AgentCheck Demo - Trace ⋅ Replay ⋅ Test your AI agents"
echo "============================================================"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Please set OPENAI_API_KEY environment variable"
    exit 1
fi

echo ""
echo "1️⃣ Running agent and capturing baseline trace..."
python demo_agent.py "Hi, I'm John Doe. Where is my order #12345?"

echo ""
echo "2️⃣ Replaying the trace with current code/model..."
agentcheck replay demo_trace.json --output replay_trace.json

echo ""  
echo "3️⃣ Comparing baseline vs replay traces..."
agentcheck diff demo_trace.json replay_trace.json

echo ""
echo "4️⃣ Asserting that response mentions customer name..."
agentcheck assert replay_trace.json --contains "John Doe"

echo ""
echo "5️⃣ Asserting cost is reasonable..."
agentcheck assert replay_trace.json --max-cost 0.01

echo ""
echo "✅ Demo completed successfully!"
echo ""
echo "Try modifying the SYSTEM_PROMPT in demo_agent.py and re-running this script"
echo "to see how agentcheck detects differences in agent behavior." 