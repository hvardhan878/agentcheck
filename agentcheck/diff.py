"""Diff engine for comparing traces."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils import load_trace

console = Console()


class TraceDiffer:
    """Engine for comparing and diffing traces."""
    
    def __init__(self) -> None:
        """Initialize the trace differ."""
        pass
    
    def diff_traces(
        self,
        trace_a_file: Path | str,
        trace_b_file: Path | str,
        output_file: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Compare two trace files and show differences.
        
        Args:
            trace_a_file: Path to first trace file (baseline)
            trace_b_file: Path to second trace file (comparison)
            output_file: Optional file to save diff results
            
        Returns:
            Diff results as a dictionary
        """
        trace_a = load_trace(trace_a_file)
        trace_b = load_trace(trace_b_file)
        
        diff_results = self._compare_traces(trace_a, trace_b)
        diff_results["trace_a_file"] = str(trace_a_file)
        diff_results["trace_b_file"] = str(trace_b_file)
        
        # Display the diff
        self._display_diff(diff_results)
        
        # Save diff results if requested
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(diff_results, f, indent=2, ensure_ascii=False)
        
        return diff_results
    
    def _compare_traces(self, trace_a: Dict[str, Any], trace_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two traces and return differences.
        
        Args:
            trace_a: First trace (baseline)
            trace_b: Second trace (comparison)
            
        Returns:
            Dictionary containing diff results
        """
        results = {
            "summary": {},
            "metadata_diff": {},
            "step_diffs": [],
            "cost_diff": {},
        }
        
        # Compare metadata
        results["metadata_diff"] = self._compare_metadata(
            trace_a.get("metadata", {}),
            trace_b.get("metadata", {}),
        )
        
        # Compare steps
        steps_a = trace_a.get("steps", [])
        steps_b = trace_b.get("steps", [])
        
        results["step_diffs"] = self._compare_steps(steps_a, steps_b)
        
        # Compare costs
        cost_a = trace_a.get("metadata", {}).get("total_cost", 0)
        cost_b = trace_b.get("metadata", {}).get("total_cost", 0)
        results["cost_diff"] = {
            "baseline": cost_a,
            "comparison": cost_b,
            "delta": cost_b - cost_a,
            "percent_change": ((cost_b - cost_a) / cost_a * 100) if cost_a > 0 else 0,
        }
        
        # Summary statistics
        results["summary"] = {
            "steps_added": len([d for d in results["step_diffs"] if d["type"] == "added"]),
            "steps_removed": len([d for d in results["step_diffs"] if d["type"] == "removed"]),
            "steps_modified": len([d for d in results["step_diffs"] if d["type"] == "modified"]),
            "steps_unchanged": len([d for d in results["step_diffs"] if d["type"] == "unchanged"]),
            "total_steps_a": len(steps_a),
            "total_steps_b": len(steps_b),
        }
        
        return results
    
    def _compare_metadata(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metadata sections."""
        diff = {"added": {}, "removed": {}, "modified": {}}
        
        all_keys = set(meta_a.keys()) | set(meta_b.keys())
        
        for key in all_keys:
            if key not in meta_a:
                diff["added"][key] = meta_b[key]
            elif key not in meta_b:
                diff["removed"][key] = meta_a[key]
            elif meta_a[key] != meta_b[key]:
                diff["modified"][key] = {"from": meta_a[key], "to": meta_b[key]}
        
        return diff
    
    def _compare_steps(
        self, 
        steps_a: List[Dict[str, Any]], 
        steps_b: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compare step arrays."""
        diffs = []
        max_steps = max(len(steps_a), len(steps_b))
        
        for i in range(max_steps):
            if i >= len(steps_a):
                diffs.append({
                    "type": "added",
                    "index": i,
                    "step": steps_b[i],
                })
            elif i >= len(steps_b):
                diffs.append({
                    "type": "removed", 
                    "index": i,
                    "step": steps_a[i],
                })
            else:
                step_a = steps_a[i]
                step_b = steps_b[i]
                
                if step_a == step_b:
                    diffs.append({
                        "type": "unchanged",
                        "index": i,
                        "step": step_a,
                    })
                else:
                    diffs.append({
                        "type": "modified",
                        "index": i,
                        "step_a": step_a,
                        "step_b": step_b,
                        "changes": self._find_step_changes(step_a, step_b),
                    })
        
        return diffs
    
    def _find_step_changes(self, step_a: Dict[str, Any], step_b: Dict[str, Any]) -> Dict[str, Any]:
        """Find specific changes between two steps."""
        changes = {}
        
        # Compare outputs
        output_a = step_a.get("output", {})
        output_b = step_b.get("output", {})
        
        if output_a.get("content") != output_b.get("content"):
            changes["content"] = {
                "from": output_a.get("content", ""),
                "to": output_b.get("content", ""),
            }
        
        # Compare costs
        cost_a = output_a.get("cost", 0)
        cost_b = output_b.get("cost", 0)
        if cost_a != cost_b:
            changes["cost"] = {"from": cost_a, "to": cost_b, "delta": cost_b - cost_a}
        
        # Compare usage
        usage_a = output_a.get("usage", {})
        usage_b = output_b.get("usage", {})
        if usage_a != usage_b:
            changes["usage"] = {"from": usage_a, "to": usage_b}
        
        return changes
    
    def _display_diff(self, diff_results: Dict[str, Any]) -> None:
        """Display diff results with rich formatting."""
        console.print()
        console.print(Panel.fit("🔍 Trace Comparison Results", style="bold blue"))
        
        # Summary table
        summary = diff_results["summary"]
        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("Steps Added", str(summary["steps_added"]))
        table.add_row("Steps Removed", str(summary["steps_removed"]))
        table.add_row("Steps Modified", str(summary["steps_modified"]))
        table.add_row("Steps Unchanged", str(summary["steps_unchanged"]))
        
        console.print(table)
        console.print()
        
        # Cost diff
        cost_diff = diff_results["cost_diff"]
        if cost_diff["delta"] != 0:
            cost_panel = Panel(
                f"Baseline: ${cost_diff['baseline']:.4f}\n"
                f"Comparison: ${cost_diff['comparison']:.4f}\n"
                f"Delta: ${cost_diff['delta']:+.4f} ({cost_diff['percent_change']:+.1f}%)",
                title="💰 Cost Changes",
                style="yellow",
            )
            console.print(cost_panel)
            console.print()
        
        # Step changes
        step_diffs = diff_results["step_diffs"]
        modified_steps = [d for d in step_diffs if d["type"] == "modified"]
        
        if modified_steps:
            console.print(Panel.fit("📝 Modified Steps", style="yellow"))
            for diff in modified_steps:
                self._display_step_diff(diff)
        
        # Added/removed steps
        added_steps = [d for d in step_diffs if d["type"] == "added"]
        removed_steps = [d for d in step_diffs if d["type"] == "removed"]
        
        if added_steps:
            console.print(Panel.fit(f"➕ {len(added_steps)} Added Steps", style="green"))
        
        if removed_steps:
            console.print(Panel.fit(f"➖ {len(removed_steps)} Removed Steps", style="red"))
    
    def _display_step_diff(self, diff: Dict[str, Any]) -> None:
        """Display a single step diff."""
        changes = diff.get("changes", {})
        
        console.print(f"  Step {diff['index'] + 1}:")
        
        for change_type, change_data in changes.items():
            if change_type == "content":
                console.print(f"    Content changed:")
                console.print(f"      - {change_data['from'][:100]}...")
                console.print(f"      + {change_data['to'][:100]}...")
            elif change_type == "cost":
                delta = change_data["delta"]
                console.print(f"    Cost: ${change_data['from']:.4f} → ${change_data['to']:.4f} ({delta:+.4f})")
            elif change_type == "usage":
                console.print(f"    Token usage changed")
        
        console.print()


def diff_traces(
    trace_a_file: Path | str,
    trace_b_file: Path | str,
    output_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Compare two trace files (convenience function).
    
    Args:
        trace_a_file: Path to first trace file (baseline)
        trace_b_file: Path to second trace file (comparison)
        output_file: Optional file to save diff results
        
    Returns:
        Diff results
    """
    differ = TraceDiffer()
    return differ.diff_traces(trace_a_file, trace_b_file, output_file) 