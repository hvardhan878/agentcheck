# AgentCheck Demo Files

This directory contains demonstration scripts showing how to use AgentCheck for AI agent testing and monitoring.

## Core Demo Files

### `demo_deterministic.py`
**Deterministic Replay Testing Demo**
- Shows how to test non-deterministic AI agents for behavioral consistency
- Creates baseline behavioral patterns from multiple runs
- Tests current agent behavior against established baselines
- Generates real LLM call data and proper trace files
- **Run this first** to establish baselines for other tests

### `test_behavioral_changes.py`
**Behavioral Change Detection Demo**
- Demonstrates how deterministic testing detects when agent behavior changes
- Uses a dramatically different agent (multiple LLM calls, different prompts)
- Shows real test failures with detailed behavioral analysis
- **Run after `demo_deterministic.py`** to see failure detection in action

### `demo_agent_with_tracing.py`
**Basic Agent Tracing Demo**
- Simple example of using AgentCheck's tracing features
- Shows how to record LLM calls, costs, and performance metrics
- Creates trace files for analytics dashboard

### `demo_agent.py`
**Basic Agent Example**
- Simple AI agent implementation
- Good starting point for understanding agent structure

## Generated Data

### `baselines/`
- Contains baseline behavioral patterns for deterministic testing
- Includes trace files from multiple runs and behavioral signatures
- Used by deterministic replay testing system

### `traces/`
- Contains individual trace files from agent runs
- Used by the analytics dashboard
- Shows detailed execution data including costs and performance

## Usage

1. **Start with basic tracing**: `python demo_agent_with_tracing.py`
2. **Set up deterministic testing**: `python demo_deterministic.py`
3. **Test behavioral changes**: `python test_behavioral_changes.py`
4. **View analytics**: Run the dashboard from the main directory

## Requirements

- OpenAI API key set in environment (`OPENAI_API_KEY`)
- AgentCheck installed (`pip install -e .` from project root)
- Python 3.8+

## What You'll See

- **Real LLM calls** with actual costs and usage data
- **Behavioral consistency testing** that works with non-deterministic outputs
- **Failure detection** when agent behavior changes significantly
- **Rich analytics** showing performance trends and patterns 