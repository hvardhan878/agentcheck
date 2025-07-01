"""Tests for trace functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentcheck import Trace, trace
from agentcheck.utils import load_trace


def test_trace_context_manager():
    """Test basic trace context manager functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        
        with Trace(output=trace_file) as t:
            t.add_step(
                "test_step",
                {"input": "test"},
                {"output": "result"},
            )
        
        # Check trace file was created
        assert trace_file.exists()
        
        # Load and validate trace
        trace_data = load_trace(trace_file)
        assert trace_data["trace_id"]
        assert trace_data["version"] == "1.0"
        assert len(trace_data["steps"]) == 1
        assert trace_data["steps"][0]["type"] == "test_step"


def test_trace_decorator():
    """Test trace decorator functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "decorator_trace.json"
        
        @trace(output=trace_file)
        def test_function(x: int) -> int:
            return x * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check trace file was created
        assert trace_file.exists()
        
        # Load and validate trace
        trace_data = load_trace(trace_file)
        assert trace_data["trace_id"]
        assert len(trace_data["steps"]) == 1
        assert trace_data["steps"][0]["type"] == "function_call"
        assert "test_function" in trace_data["metadata"]["function_name"]


def test_llm_call_tracing():
    """Test LLM call tracing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "llm_trace.json"
        
        with Trace(output=trace_file) as t:
            messages = [{"role": "user", "content": "Hello"}]
            response = {
                "content": "Hi there!",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
            
            t.add_llm_call(messages, response, "gpt-4o-mini")
        
        # Load and validate trace
        trace_data = load_trace(trace_file)
        assert len(trace_data["steps"]) == 1
        
        step = trace_data["steps"][0]
        assert step["type"] == "llm_call"
        assert step["input"]["messages"] == messages
        assert step["input"]["model"] == "gpt-4o-mini"
        assert step["output"]["content"] == "Hi there!"
        assert "cost" in step["output"]


def test_trace_with_exception():
    """Test trace behavior when exception occurs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "exception_trace.json"
        
        with pytest.raises(ValueError):
            with Trace(output=trace_file) as t:
                t.add_step("test", {"input": "test"})
                raise ValueError("Test error")
        
        # Trace should still be saved with exception info
        assert trace_file.exists()
        trace_data = load_trace(trace_file)
        assert "exception" in trace_data["metadata"]
        assert trace_data["metadata"]["exception"]["type"] == "ValueError"


def test_trace_without_output():
    """Test trace without output file."""
    with Trace() as t:
        t.add_step("test", {"input": "test"})
        # Should not raise any errors
        assert len(t.steps) == 1 