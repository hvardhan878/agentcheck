"""Tests for diff functionality."""

import json
import tempfile
from pathlib import Path

from agentcheck.diff import TraceDiffer


def create_test_trace(trace_id: str, content: str, cost: float = 0.001) -> dict:
    """Create a test trace with specified content."""
    return {
        "trace_id": trace_id,
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
                "input": {"messages": [{"role": "user", "content": "test"}]},
                "output": {"content": content, "cost": cost},
            }
        ],
    }


def test_identical_traces():
    """Test diffing identical traces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a_file = Path(tmpdir) / "trace_a.json"
        trace_b_file = Path(tmpdir) / "trace_b.json"
        
        trace_data = create_test_trace("test1", "Hello world")
        
        with open(trace_a_file, "w") as f:
            json.dump(trace_data, f)
        with open(trace_b_file, "w") as f:
            json.dump(trace_data, f)
        
        differ = TraceDiffer()
        result = differ.diff_traces(trace_a_file, trace_b_file)
        
        assert result["summary"]["steps_modified"] == 0
        assert result["summary"]["steps_unchanged"] == 1
        assert result["cost_diff"]["delta"] == 0


def test_different_content():
    """Test diffing traces with different content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a_file = Path(tmpdir) / "trace_a.json"
        trace_b_file = Path(tmpdir) / "trace_b.json"
        
        trace_a = create_test_trace("test1", "Hello world", 0.001)
        trace_b = create_test_trace("test2", "Goodbye world", 0.002)
        
        with open(trace_a_file, "w") as f:
            json.dump(trace_a, f)
        with open(trace_b_file, "w") as f:
            json.dump(trace_b, f)
        
        differ = TraceDiffer()
        result = differ.diff_traces(trace_a_file, trace_b_file)
        
        assert result["summary"]["steps_modified"] == 1
        assert result["summary"]["steps_unchanged"] == 0
        assert result["cost_diff"]["delta"] == 0.001


def test_added_removed_steps():
    """Test diffing traces with different number of steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a_file = Path(tmpdir) / "trace_a.json"
        trace_b_file = Path(tmpdir) / "trace_b.json"
        
        trace_a = create_test_trace("test1", "Hello")
        trace_b = create_test_trace("test2", "Hello")
        
        # Add an extra step to trace_b
        trace_b["steps"].append({
            "step_id": "step2",
            "start_time": "2024-01-01T12:00:05Z",
            "end_time": "2024-01-01T12:00:06Z",
            "type": "llm_call",
            "input": {"messages": [{"role": "user", "content": "more"}]},
            "output": {"content": "Extra step", "cost": 0.001},
        })
        
        with open(trace_a_file, "w") as f:
            json.dump(trace_a, f)
        with open(trace_b_file, "w") as f:
            json.dump(trace_b, f)
        
        differ = TraceDiffer()
        result = differ.diff_traces(trace_a_file, trace_b_file)
        
        assert result["summary"]["steps_added"] == 1
        assert result["summary"]["steps_removed"] == 0
        assert result["summary"]["steps_unchanged"] == 1 