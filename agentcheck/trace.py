"""Trace decorator and context manager for capturing agent execution."""

import functools
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from .utils import calculate_cost, generate_id, get_current_time, save_trace

F = TypeVar("F", bound=Callable[..., Any])


class Trace:
    """Context manager for tracing agent execution."""
    
    def __init__(self, output: Optional[Union[str, Path]] = None) -> None:
        """Initialize trace context manager.
        
        Args:
            output: Output file path for the trace. If None, trace is not saved.
        """
        self.output = Path(output) if output else None
        self.trace_id = generate_id()
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.steps: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
    def __enter__(self) -> "Trace":
        """Enter the trace context."""
        self.start_time = get_current_time()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the trace context and save trace if output is specified."""
        self.end_time = get_current_time()
        
        if exc_type is not None:
            # Record exception in metadata
            self.metadata["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": "".join(traceback.format_tb(exc_tb)),
            }
        
        if self.output:
            self._save_trace()
    
    def add_step(
        self,
        step_type: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a step to the trace.
        
        Args:
            step_type: Type of step (llm_call, tool_call, function_call)
            input_data: Input data for the step
            output_data: Output data from the step
            error: Error information if step failed
            
        Returns:
            The step ID
        """
        step_id = generate_id()
        start_time = get_current_time()
        
        step = {
            "step_id": step_id,
            "start_time": start_time,
            "end_time": get_current_time(),
            "type": step_type,
            "input": input_data,
        }
        
        if output_data:
            step["output"] = output_data
        if error:
            step["error"] = error
            
        self.steps.append(step)
        return step_id
    
    def add_llm_call(
        self,
        messages: List[Dict[str, str]],
        response: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o-mini",
        error: Optional[Exception] = None,
    ) -> str:
        """Add an LLM call step to the trace.
        
        Args:
            messages: Chat messages sent to the LLM
            response: Response from the LLM
            model: Model used for the call
            error: Exception if the call failed
            
        Returns:
            The step ID
        """
        input_data = {
            "messages": messages,
            "model": model,
        }
        
        output_data = None
        error_data = None
        
        if error:
            error_data = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        elif response:
            output_data = {
                "content": response.get("content", ""),
                "model": model,
            }
            
            if "usage" in response:
                output_data["usage"] = response["usage"]
                output_data["cost"] = calculate_cost(response["usage"], model)
        
        return self.add_step("llm_call", input_data, output_data, error_data)
    
    def _save_trace(self) -> None:
        """Save the trace to the output file."""
        if not self.output:
            return
            
        trace_data = {
            "trace_id": self.trace_id,
            "version": "1.0",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "steps": self.steps,
        }
        
        # Calculate total cost
        total_cost = sum(
            step.get("output", {}).get("cost", 0)
            for step in self.steps
        )
        if total_cost > 0:
            trace_data["metadata"]["total_cost"] = total_cost
        
        save_trace(trace_data, self.output)


def trace(output: Optional[Union[str, Path]] = None) -> Callable[[F], F]:
    """Decorator to trace function execution.
    
    Args:
        output: Output file path for the trace
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Trace(output) as t:
                # Add function call metadata
                t.metadata["function_name"] = func.__name__
                t.metadata["function_module"] = func.__module__
                
                try:
                    # Record function call as a step
                    step_id = t.add_step(
                        "function_call",
                        {
                            "function_name": func.__name__,
                            "arguments": {"args": args, "kwargs": kwargs},
                        }
                    )
                    
                    result = func(*args, **kwargs)
                    
                    # Update the step with the result
                    for step in t.steps:
                        if step["step_id"] == step_id:
                            step["output"] = {"result": str(result)[:1000]}  # Truncate long results
                            step["end_time"] = get_current_time()
                            break
                    
                    return result
                    
                except Exception as e:
                    # Update the step with error information
                    for step in t.steps:
                        if step["step_id"] == step_id:
                            step["error"] = {
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": traceback.format_exc(),
                            }
                            step["end_time"] = get_current_time()
                            break
                    raise
                    
        return wrapper  # type: ignore
    return decorator


@contextmanager
def trace_openai_call(
    tracer: Trace,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
) -> Generator[None, None, None]:
    """Context manager to trace OpenAI API calls.
    
    Args:
        tracer: Active trace instance
        messages: Messages to send to OpenAI
        model: Model to use
    """
    step_start = get_current_time()
    step_id = None
    
    try:
        yield
    except Exception as e:
        step_id = tracer.add_llm_call(messages, None, model, e)
        raise
    finally:
        if step_id:
            # Update end time
            for step in tracer.steps:
                if step["step_id"] == step_id:
                    step["end_time"] = get_current_time()
                    break 