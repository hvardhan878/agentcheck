"""Assertion engine for testing trace contents."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonpath_ng
from rich.console import Console

from .utils import load_trace

console = Console()


class TraceAssertionError(Exception):
    """Exception raised when a trace assertion fails."""
    pass


class TraceAsserter:
    """Engine for making assertions about trace contents."""
    
    def __init__(self) -> None:
        """Initialize the trace asserter."""
        pass
    
    def assert_trace(
        self,
        trace_file: Path | str,
        contains: Optional[str] = None,
        not_contains: Optional[str] = None,
        jsonpath: Optional[str] = None,
        regex: Optional[str] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        exit_on_failure: bool = True,
    ) -> bool:
        """Assert conditions about a trace file.
        
        Args:
            trace_file: Path to the trace file
            contains: String that should be present in trace content
            not_contains: String that should NOT be present in trace content
            jsonpath: JSONPath expression to extract specific data
            regex: Regex pattern to match against content
            min_cost: Minimum expected cost
            max_cost: Maximum expected cost
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps
            exit_on_failure: Whether to exit process on assertion failure
            
        Returns:
            True if all assertions pass
            
        Raises:
            TraceAssertionError: If any assertion fails and exit_on_failure is False
        """
        try:
            trace_data = load_trace(trace_file)
            
            # Extract content to search in
            content = self._extract_content(trace_data, jsonpath)
            
            # Run assertions
            if contains:
                self._assert_contains(content, contains)
            
            if not_contains:
                self._assert_not_contains(content, not_contains)
            
            if regex:
                self._assert_regex(content, regex)
            
            if min_cost is not None or max_cost is not None:
                self._assert_cost_range(trace_data, min_cost, max_cost)
            
            if min_steps is not None or max_steps is not None:
                self._assert_step_count(trace_data, min_steps, max_steps)
            
            console.print("✅ All assertions passed", style="green")
            return True
            
        except (TraceAssertionError, FileNotFoundError, ValueError) as e:
            console.print(f"❌ Assertion failed: {e}", style="red")
            if exit_on_failure:
                sys.exit(1)
            raise
    
    def _extract_content(self, trace_data: Dict[str, Any], jsonpath: Optional[str] = None) -> str:
        """Extract content from trace data using JSONPath or default extraction.
        
        Args:
            trace_data: The trace data
            jsonpath: Optional JSONPath expression
            
        Returns:
            Extracted content as string
        """
        if jsonpath:
            try:
                parser = jsonpath_ng.parse(jsonpath)
                matches = parser.find(trace_data)
                if not matches:
                    raise TraceAssertionError(f"JSONPath '{jsonpath}' returned no matches")
                
                # Combine all matches into a single string
                content_parts = []
                for match in matches:
                    if isinstance(match.value, str):
                        content_parts.append(match.value)
                    else:
                        content_parts.append(json.dumps(match.value))
                
                return " ".join(content_parts)
                
            except Exception as e:
                raise TraceAssertionError(f"Error evaluating JSONPath '{jsonpath}': {e}")
        else:
            # Default: extract all content from step outputs
            content_parts = []
            
            for step in trace_data.get("steps", []):
                output = step.get("output", {})
                if "content" in output:
                    content_parts.append(str(output["content"]))
                if "result" in output:
                    content_parts.append(str(output["result"]))
            
            return " ".join(content_parts)
    
    def _assert_contains(self, content: str, target: str) -> None:
        """Assert that content contains the target string."""
        if target not in content:
            raise TraceAssertionError(f"Content does not contain '{target}'")
    
    def _assert_not_contains(self, content: str, target: str) -> None:
        """Assert that content does not contain the target string."""
        if target in content:
            raise TraceAssertionError(f"Content should not contain '{target}' but it does")
    
    def _assert_regex(self, content: str, pattern: str) -> None:
        """Assert that content matches the regex pattern."""
        try:
            if not re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                raise TraceAssertionError(f"Content does not match regex pattern '{pattern}'")
        except re.error as e:
            raise TraceAssertionError(f"Invalid regex pattern '{pattern}': {e}")
    
    def _assert_cost_range(
        self, 
        trace_data: Dict[str, Any], 
        min_cost: Optional[float], 
        max_cost: Optional[float],
    ) -> None:
        """Assert that total cost is within the specified range."""
        total_cost = trace_data.get("metadata", {}).get("total_cost", 0)
        
        if min_cost is not None and total_cost < min_cost:
            raise TraceAssertionError(f"Total cost ${total_cost:.4f} is below minimum ${min_cost:.4f}")
        
        if max_cost is not None and total_cost > max_cost:
            raise TraceAssertionError(f"Total cost ${total_cost:.4f} exceeds maximum ${max_cost:.4f}")
    
    def _assert_step_count(
        self, 
        trace_data: Dict[str, Any], 
        min_steps: Optional[int], 
        max_steps: Optional[int],
    ) -> None:
        """Assert that step count is within the specified range."""
        step_count = len(trace_data.get("steps", []))
        
        if min_steps is not None and step_count < min_steps:
            raise TraceAssertionError(f"Step count {step_count} is below minimum {min_steps}")
        
        if max_steps is not None and step_count > max_steps:
            raise TraceAssertionError(f"Step count {step_count} exceeds maximum {max_steps}")


def assert_trace(
    trace_file: Path | str,
    contains: Optional[str] = None,
    not_contains: Optional[str] = None,
    jsonpath: Optional[str] = None,
    regex: Optional[str] = None,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    min_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    exit_on_failure: bool = True,
) -> bool:
    """Assert conditions about a trace file (convenience function).
    
    Args:
        trace_file: Path to the trace file
        contains: String that should be present in trace content
        not_contains: String that should NOT be present in trace content
        jsonpath: JSONPath expression to extract specific data
        regex: Regex pattern to match against content
        min_cost: Minimum expected cost
        max_cost: Maximum expected cost
        min_steps: Minimum number of steps
        max_steps: Maximum number of steps
        exit_on_failure: Whether to exit process on assertion failure
        
    Returns:
        True if all assertions pass
    """
    asserter = TraceAsserter()
    return asserter.assert_trace(
        trace_file=trace_file,
        contains=contains,
        not_contains=not_contains,
        jsonpath=jsonpath,
        regex=regex,
        min_cost=min_cost,
        max_cost=max_cost,
        min_steps=min_steps,
        max_steps=max_steps,
        exit_on_failure=exit_on_failure,
    ) 