"""Utility functions for agentcheck."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import jsonschema
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def get_current_time() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def load_trace(file_path: Path | str) -> Dict[str, Any]:
    """Load and validate a trace file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Trace file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    
    # Validate against schema
    validate_trace(trace_data)
    return trace_data


def save_trace(trace_data: Dict[str, Any], file_path: Path | str) -> None:
    """Save trace data to file with validation."""
    validate_trace(trace_data)
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)


def validate_trace(trace_data: Dict[str, Any]) -> None:
    """Validate trace data against schema."""
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(trace_data, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid trace format: {e.message}") from e


def pretty_print_json(data: Dict[str, Any]) -> None:
    """Pretty print JSON data with syntax highlighting."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(syntax)


def calculate_cost(usage: Dict[str, int], model: str = "gpt-4o-mini") -> float:
    """Calculate estimated cost based on token usage."""
    # Simplified cost calculation - in practice you'd have a more comprehensive model
    cost_per_1k = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    }
    
    if model not in cost_per_1k:
        model = "gpt-4o-mini"  # fallback
    
    rates = cost_per_1k[model]
    prompt_cost = (usage.get("prompt_tokens", 0) / 1000) * rates["input"]
    completion_cost = (usage.get("completion_tokens", 0) / 1000) * rates["output"]
    
    return prompt_cost + completion_cost 