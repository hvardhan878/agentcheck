"""Tests for assertion functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from agentcheck.asserts import TraceAsserter, TraceAssertionError


def create_test_trace(content: str, cost: float = 0.005) -> dict:
    """Create a test trace with specified content and cost."""
    return {
        "trace_id": "test-trace",
        "version": "1.0",
        "start_time": "2024-01-01T12:00:00Z",
        "end_time": "2024-01-01T12:00:05Z",
        "metadata": {"total_cost": cost},
        "steps": [
            {
                "step_id": "step1",
                "start_time": "2024-01-01T12:00:01Z",
                "end_time": "2024-01-01T12:00:04Z",
                "type": "llm_call",
                "input": {"messages": [{"role": "user", "content": "Hello John"}]},
                "output": {"content": content, "cost": cost},
            }
        ],
    }


def test_assert_contains_success():
    """Test successful contains assertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Hello John, how can I help you?")
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        result = asserter.assert_trace(trace_file, contains="John", exit_on_failure=False)
        assert result is True


def test_assert_contains_failure():
    """Test failed contains assertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Hello there, how can I help you?")
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        with pytest.raises(TraceAssertionError, match="does not contain 'John'"):
            asserter.assert_trace(trace_file, contains="John", exit_on_failure=False)


def test_assert_not_contains():
    """Test not_contains assertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Hello there, how can I help you?")
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        result = asserter.assert_trace(trace_file, not_contains="John", exit_on_failure=False)
        assert result is True
        
        # Should fail when string is present
        with pytest.raises(TraceAssertionError, match="should not contain"):
            asserter.assert_trace(trace_file, not_contains="Hello", exit_on_failure=False)


def test_assert_cost_range():
    """Test cost range assertions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Hello", cost=0.005)
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        
        # Should pass
        result = asserter.assert_trace(
            trace_file, min_cost=0.001, max_cost=0.010, exit_on_failure=False
        )
        assert result is True
        
        # Should fail - cost too high
        with pytest.raises(TraceAssertionError, match="exceeds maximum"):
            asserter.assert_trace(trace_file, max_cost=0.001, exit_on_failure=False)
        
        # Should fail - cost too low
        with pytest.raises(TraceAssertionError, match="below minimum"):
            asserter.assert_trace(trace_file, min_cost=0.010, exit_on_failure=False)


def test_assert_step_count():
    """Test step count assertions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Hello")
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        
        # Should pass - has 1 step
        result = asserter.assert_trace(
            trace_file, min_steps=1, max_steps=2, exit_on_failure=False
        )
        assert result is True
        
        # Should fail - too many steps required
        with pytest.raises(TraceAssertionError, match="below minimum"):
            asserter.assert_trace(trace_file, min_steps=5, exit_on_failure=False)


def test_assert_regex():
    """Test regex assertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "test_trace.json"
        trace_data = create_test_trace("Order #12345 is ready")
        
        with open(trace_file, "w") as f:
            json.dump(trace_data, f)
        
        asserter = TraceAsserter()
        
        # Should pass - matches order pattern
        result = asserter.assert_trace(
            trace_file, regex=r"Order #\d+", exit_on_failure=False
        )
        assert result is True
        
        # Should fail - doesn't match pattern
        with pytest.raises(TraceAssertionError, match="does not match regex"):
            asserter.assert_trace(trace_file, regex=r"Invoice #\d+", exit_on_failure=False) 