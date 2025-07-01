"""Replay engine for re-executing traces."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .trace import Trace
from .utils import calculate_cost, get_current_time, load_trace, save_trace

console = Console()


class ReplayEngine:
    """Engine for replaying agent traces."""
    
    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        """Initialize replay engine.
        
        Args:
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.client = openai.OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def replay_trace(
        self,
        trace_file: Path | str,
        output_file: Optional[Path | str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replay a trace file.
        
        Args:
            trace_file: Path to the original trace file
            output_file: Path to save the new trace. If None, uses original name with _replay suffix
            model_override: Override model for all LLM calls
            
        Returns:
            The new trace data
        """
        original_trace = load_trace(trace_file)
        
        if output_file is None:
            trace_path = Path(trace_file)
            output_file = trace_path.parent / f"{trace_path.stem}_replay{trace_path.suffix}"
        
        with Trace(output=output_file) as new_trace:
            new_trace.metadata.update({
                "original_trace_id": original_trace["trace_id"],
                "replay_mode": True,
                "model_override": model_override,
            })
            
            console.print(f"🔄 Replaying trace: {trace_file}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Replaying steps...", total=len(original_trace["steps"]))
                
                for i, original_step in enumerate(original_trace["steps"]):
                    progress.update(task, advance=1, description=f"Step {i+1}/{len(original_trace['steps'])}")
                    
                    if original_step["type"] == "llm_call":
                        self._replay_llm_call(new_trace, original_step, model_override)
                    elif original_step["type"] == "function_call":
                        self._replay_function_call(new_trace, original_step)
                    else:
                        # For other step types, just copy the original
                        new_trace.steps.append(original_step.copy())
        
        console.print(f"✅ Replay complete: {output_file}")
        return load_trace(output_file)
    
    def _replay_llm_call(
        self,
        new_trace: Trace,
        original_step: Dict[str, Any],
        model_override: Optional[str] = None,
    ) -> None:
        """Replay an LLM call step.
        
        Args:
            new_trace: New trace to add the step to
            original_step: Original step data
            model_override: Override model for the call
        """
        input_data = original_step["input"]
        messages = input_data["messages"]
        model = model_override or input_data.get("model", "gpt-4o-mini")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # Deterministic for testing
            )
            
            # Convert response to dict format
            response_dict = {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else {},
            }
            
            new_trace.add_llm_call(messages, response_dict, model)
            
        except Exception as e:
            console.print(f"⚠️  Error replaying LLM call: {e}")
            new_trace.add_llm_call(messages, None, model, e)
    
    def _replay_function_call(
        self,
        new_trace: Trace,
        original_step: Dict[str, Any],
    ) -> None:
        """Replay a function call step.
        
        Args:
            new_trace: New trace to add the step to
            original_step: Original step data
        """
        # For function calls, we can't actually re-execute them without the original code
        # So we just copy the step and mark it as non-replayed
        step_copy = original_step.copy()
        step_copy["metadata"] = step_copy.get("metadata", {})
        step_copy["metadata"]["replayed"] = False
        step_copy["metadata"]["reason"] = "Function calls cannot be replayed without original code"
        
        new_trace.steps.append(step_copy)


def replay_trace(
    trace_file: Path | str,
    output_file: Optional[Path | str] = None,
    model_override: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay a trace file (convenience function).
    
    Args:
        trace_file: Path to the original trace file
        output_file: Path to save the new trace
        model_override: Override model for all LLM calls
        openai_api_key: OpenAI API key
        
    Returns:
        The new trace data
    """
    engine = ReplayEngine(openai_api_key)
    return engine.replay_trace(trace_file, output_file, model_override) 