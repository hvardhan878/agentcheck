# agentcheck

> **agentcheck: Trace ⋅ Replay ⋅ Test your AI agents like real software.**

[![PyPI version](https://badge.fury.io/py/agentcheck.svg)](https://badge.fury.io/py/agentcheck)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AgentCheck is a minimal but complete toolkit for **tracing**, **replaying**, **diffing**, and **testing** AI agent executions. Think of it as version control and testing for your AI agents.

## 🚀 Install

```bash
pip install agentcheck
```

## ⚡ Quickstart Demo

```bash
export OPENAI_API_KEY=sk-...

# 1️⃣ Capture baseline trace
python demo/demo_agent.py --output baseline.json

# 2️⃣ Modify the prompt inside demo_agent.py (e.g. change tone)
# 3️⃣ Replay with new code/model  
agentcheck replay baseline.json --output new.json

# 4️⃣ See what changed
agentcheck diff baseline.json new.json

# 5️⃣ Assert the new output still mentions the user's name
agentcheck assert new.json --contains "John Doe"

# 🆕 6️⃣ Test deterministic behavior
python demo/demo_deterministic.py
```

Or run the complete demo:

```bash
cd demo && ./demo_run.sh
```

## 🎯 Features

| Feature | Description | CLI Command | Python API |
|---------|-------------|-------------|------------|
| **Trace** | Capture agent execution (prompts, outputs, costs, timing) | `agentcheck trace <command>` | `@agentcheck.trace()` |
| **Replay** | Re-run trace against current code/model | `agentcheck replay trace.json` | `agentcheck.replay_trace()` |
| **Diff** | Compare traces and highlight changes | `agentcheck diff trace_a.json trace_b.json` | `agentcheck.diff_traces()` |
| **Assert** | Test trace contents (CI-friendly) | `agentcheck assert trace.json --contains "foo"` | `agentcheck.assert_trace()` |
| **🆕 Deterministic Testing** | Test behavioral consistency of non-deterministic agents | *(Python API only)* | `@agentcheck.deterministic_replay()` |
| **🆕 Analytics Dashboard** | Beautiful web GUI for trace analysis and testing insights | `python launch_dashboard.py` | *Web interface* |

## 📖 Usage

### Tracing with Decorator

```python
import agentcheck
import openai

@agentcheck.trace(output="my_trace.json")
def my_agent(user_input: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# Automatically traces execution and saves to my_trace.json
result = my_agent("Hello, world!")
```

### 🆕 Deterministic Replay Testing

**The Problem**: AI agents are non-deterministic - they produce different outputs for identical inputs, making traditional testing impossible.

**The Solution**: AgentCheck's deterministic replay testing learns your agent's behavioral patterns and detects when behavior changes unexpectedly.

```python
import agentcheck
import openai

@agentcheck.deterministic_replay(
    consistency_threshold=0.8,  # 80% behavioral consistency required
    baseline_runs=5,           # Run 5 times to establish baseline
    baseline_name="my_agent"   # Name for this baseline
)
def my_agent(user_input: str) -> str:
    with agentcheck.trace() as trace:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.7  # Non-deterministic!
        )
        
        # Record the LLM call
        trace.add_llm_call(
            messages=[{"role": "user", "content": user_input}],
            response={"content": response.choices[0].message.content},
            model="gpt-4o-mini"
        )
        
        return response.choices[0].message.content

# Step 1: Establish behavioral baseline
replayer = my_agent._deterministic_replayer
test_inputs = ["What is Python?", "How do I install packages?"]

replayer.establish_baseline(
    agent_func=my_agent,
    test_inputs=test_inputs,
    baseline_name="my_agent"
)

# Step 2: Test current agent against baseline
failures = replayer.test_consistency(
    agent_func=my_agent,
    test_inputs=test_inputs,
    baseline_name="my_agent"
)

if failures:
    print(f"❌ {len(failures)} tests failed - agent behavior changed!")
    for failure in failures:
        print(f"Input: {failure.input_data}")
        print(f"Consistency Score: {failure.consistency_score:.3f}")
else:
    print("✅ All tests passed - agent behavior is consistent!")
```

**What it detects:**
- Changes in reasoning patterns
- Different tool usage sequences  
- Altered response structures
- Performance regressions
- Error rate changes

**Perfect for:**
- Regression testing after prompt changes
- Model version upgrades
- Code refactoring validation
- CI/CD pipeline integration

### 🆕 Analytics Dashboard

Get beautiful insights into your agent performance with the built-in web dashboard:

```bash
# Launch the dashboard
python launch_dashboard.py

# Or manually with streamlit
pip install streamlit plotly pandas numpy
streamlit run agentcheck_dashboard.py
```

**Dashboard Features:**
- **📊 Overview**: Key metrics, traces over time, model usage distribution
- **🔍 Trace Analysis**: Detailed step-by-step execution analysis 
- **🧪 Deterministic Testing**: Baseline management and consistency trends
- **💰 Cost Analysis**: Cost breakdowns by model and time periods

**What you can track:**
- Total traces and execution costs
- Error rates and failure patterns  
- LLM model usage and performance
- Behavioral consistency trends
- Cost optimization opportunities

The dashboard automatically loads data from your `traces/` and `baselines/` directories and provides real-time analytics as you develop and test your agents.

### Tracing with Context Manager

```python
import agentcheck

with agentcheck.Trace(output="trace.json") as t:
    # Your agent code here
    messages = [{"role": "user", "content": "Hello"}]
    
    # Manually add LLM calls to trace
    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    
    t.add_llm_call(
        messages=messages,
        response={"content": response.choices[0].message.content, "usage": response.usage},
        model="gpt-4o-mini"
    )
```

### CLI Commands

```bash
# Trace a Python script
agentcheck trace "python my_agent.py" --output trace.json

# Replay a trace with a different model
agentcheck replay trace.json --model gpt-4 --output new_trace.json

# Compare two traces
agentcheck diff baseline.json new_trace.json

# Assert trace contains expected content
agentcheck assert trace.json --contains "expected output"

# Assert with JSONPath
agentcheck assert trace.json --jsonpath "$.steps[-1].output.content" --contains "John"

# Assert cost and step constraints
agentcheck assert trace.json --max-cost 0.05 --min-steps 1 --max-steps 10

# Pretty-print a trace
agentcheck show trace.json
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Your Agent    │───▶│ agentcheck   │───▶│  trace.json     │
│                 │    │   tracer     │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
                    │     replay      │   │      diff       │   │     assert      │
                    │   (re-execute)  │   │   (compare)     │   │    (test)       │
                    └─────────────────┘   └─────────────────┘   └─────────────────┘
                                                    │
                                          ┌─────────────────────┐
                                          │ 🆕 deterministic    │
                                          │ behavioral testing  │
                                          └─────────────────────┘
```

## 📋 Trace Format

AgentCheck uses a standardized JSON schema for traces:

```json
{
  "trace_id": "uuid",
  "version": "1.0", 
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:00:05Z",
  "metadata": {
    "total_cost": 0.0023,
    "function_name": "my_agent"
  },
  "steps": [
    {
      "step_id": "uuid",
      "type": "llm_call",
      "start_time": "2024-01-01T12:00:01Z", 
      "end_time": "2024-01-01T12:00:04Z",
      "input": {
        "messages": [...],
        "model": "gpt-4o-mini"
      },
      "output": {
        "content": "Agent response...",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "cost": 0.0023
      }
    }
  ]
}
```

## 🧪 Testing & CI Integration

AgentCheck is designed for CI/CD pipelines:

```bash
# Traditional trace testing
agentcheck replay baseline_trace.json --output ci_trace.json
agentcheck assert ci_trace.json --contains "expected behavior" --max-cost 0.10

# 🆕 Deterministic behavioral testing
python -c "
import agentcheck
from my_agent import my_agent

replayer = my_agent._deterministic_replayer
test_inputs = ['test1', 'test2', 'test3']

failures = replayer.test_consistency(
    agent_func=my_agent,
    test_inputs=test_inputs,
    baseline_name='production'
)

if failures:
    print(f'❌ {len(failures)} behavioral consistency tests failed')
    exit(1)
else:
    print('✅ All behavioral tests passed')
    exit(0)
"

# Exit codes
# 0 = success
# 1 = assertion failed or error
```

## 🛠️ Development

```bash
# Install in development mode
git clone https://github.com/agentcheck/agentcheck
cd agentcheck
pip install -e ".[dev]"

# Run tests
pytest

# Format code  
ruff format .

# Type check
mypy agentcheck/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built for the era of AI agents** 🤖✨ 