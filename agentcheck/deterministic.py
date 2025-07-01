"""Deterministic replay testing for non-deterministic AI agents."""

import functools
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .trace import Trace
from .utils import generate_id, get_current_time, load_trace, save_trace

F = TypeVar("F", bound=Callable[..., Any])


class BehavioralSignature:
    """Represents the behavioral pattern of an agent execution."""
    
    def __init__(self, traces: List[Dict[str, Any]]):
        """Initialize behavioral signature from multiple traces.
        
        Args:
            traces: List of trace dictionaries from multiple runs
        """
        self.traces = traces
        self.signature = self._extract_signature()
    
    def _extract_signature(self) -> Dict[str, Any]:
        """Extract behavioral signature from traces."""
        if not self.traces:
            return {}
        
        # Extract key behavioral patterns
        signature = {
            "step_count": self._analyze_step_counts(),
            "step_types": self._analyze_step_types(),
            "decision_patterns": self._analyze_decision_patterns(),
            "error_patterns": self._analyze_error_patterns(),
            "performance_metrics": self._analyze_performance(),
        }
        
        return signature
    
    def _analyze_step_counts(self) -> Dict[str, Any]:
        """Analyze the number of steps across runs."""
        step_counts = [len(trace.get("steps", [])) for trace in self.traces]
        return {
            "mean": statistics.mean(step_counts) if step_counts else 0,
            "std": statistics.stdev(step_counts) if len(step_counts) > 1 else 0,
            "min": min(step_counts) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
        }
    
    def _analyze_step_types(self) -> Dict[str, Any]:
        """Analyze the types of steps taken."""
        all_step_types = []
        for trace in self.traces:
            step_types = [step.get("type", "unknown") for step in trace.get("steps", [])]
            all_step_types.append(step_types)
        
        # Find most common step sequence pattern
        if all_step_types:
            # Convert to string representation for pattern matching
            patterns = ["|".join(types) for types in all_step_types]
            most_common = max(set(patterns), key=patterns.count) if patterns else ""
            consistency = patterns.count(most_common) / len(patterns) if patterns else 0
        else:
            most_common = ""
            consistency = 0
        
        return {
            "common_pattern": most_common,
            "pattern_consistency": consistency,
            "unique_patterns": len(set(patterns)) if 'patterns' in locals() else 0,
        }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze decision-making patterns in LLM calls."""
        llm_calls = []
        for trace in self.traces:
            for step in trace.get("steps", []):
                if step.get("type") == "llm_call":
                    llm_calls.append(step)
        
        if not llm_calls:
            return {"llm_call_count": 0}
        
        # Analyze response patterns
        response_lengths = []
        models_used = []
        
        for call in llm_calls:
            output = call.get("output", {})
            content = output.get("content", "")
            response_lengths.append(len(content))
            models_used.append(output.get("model", "unknown"))
        
        return {
            "llm_call_count": len(llm_calls),
            "avg_response_length": statistics.mean(response_lengths) if response_lengths else 0,
            "models_used": list(set(models_used)),
            "response_length_consistency": 1 - (statistics.stdev(response_lengths) / statistics.mean(response_lengths)) if len(response_lengths) > 1 and statistics.mean(response_lengths) > 0 else 1,
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns across runs."""
        error_count = 0
        error_types = []
        
        for trace in self.traces:
            if trace.get("metadata", {}).get("exception"):
                error_count += 1
                error_types.append(trace["metadata"]["exception"]["type"])
            
            for step in trace.get("steps", []):
                if step.get("error"):
                    error_count += 1
                    error_types.append(step["error"]["type"])
        
        return {
            "error_rate": error_count / len(self.traces) if self.traces else 0,
            "error_types": list(set(error_types)),
            "total_errors": error_count,
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        costs = []
        durations = []
        
        for trace in self.traces:
            # Extract cost
            cost = trace.get("metadata", {}).get("total_cost", 0)
            if cost > 0:
                costs.append(cost)
            
            # Calculate duration
            start = trace.get("start_time")
            end = trace.get("end_time")
            if start and end:
                # Simple duration calculation (would need proper datetime parsing in production)
                durations.append(1.0)  # Placeholder
        
        return {
            "avg_cost": statistics.mean(costs) if costs else 0,
            "cost_consistency": 1 - (statistics.stdev(costs) / statistics.mean(costs)) if len(costs) > 1 and statistics.mean(costs) > 0 else 1,
            "avg_duration": statistics.mean(durations) if durations else 0,
            "runs_analyzed": len(self.traces),
        }


class ConsistencyScore:
    """Calculates consistency scores between behavioral signatures."""
    
    @staticmethod
    def calculate(baseline: BehavioralSignature, current: BehavioralSignature) -> float:
        """Calculate consistency score between two behavioral signatures.
        
        Args:
            baseline: Baseline behavioral signature
            current: Current behavioral signature to compare
            
        Returns:
            Consistency score between 0 and 1 (1 = perfectly consistent)
        """
        baseline_sig = baseline.signature
        current_sig = current.signature
        
        scores = []
        
        # Step count consistency
        step_score = ConsistencyScore._compare_step_counts(
            baseline_sig.get("step_count", {}),
            current_sig.get("step_count", {})
        )
        scores.append(step_score)
        
        # Step type pattern consistency
        pattern_score = ConsistencyScore._compare_step_patterns(
            baseline_sig.get("step_types", {}),
            current_sig.get("step_types", {})
        )
        scores.append(pattern_score)
        
        # Decision pattern consistency
        decision_score = ConsistencyScore._compare_decision_patterns(
            baseline_sig.get("decision_patterns", {}),
            current_sig.get("decision_patterns", {})
        )
        scores.append(decision_score)
        
        # Error pattern consistency
        error_score = ConsistencyScore._compare_error_patterns(
            baseline_sig.get("error_patterns", {}),
            current_sig.get("error_patterns", {})
        )
        scores.append(error_score)
        
        # Weighted average (you can adjust weights based on importance)
        weights = [0.3, 0.3, 0.25, 0.15]
        return sum(score * weight for score, weight in zip(scores, weights))
    
    @staticmethod
    def _compare_step_counts(baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Compare step count patterns."""
        if not baseline or not current:
            return 0.0
        
        baseline_mean = baseline.get("mean", 0)
        current_mean = current.get("mean", 0)
        
        if baseline_mean == 0:
            return 1.0 if current_mean == 0 else 0.0
        
        # Calculate relative difference
        diff = abs(baseline_mean - current_mean) / baseline_mean
        return max(0, 1 - diff)
    
    @staticmethod
    def _compare_step_patterns(baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Compare step type patterns."""
        baseline_pattern = baseline.get("common_pattern", "")
        current_pattern = current.get("common_pattern", "")
        
        if baseline_pattern == current_pattern:
            return 1.0
        elif not baseline_pattern and not current_pattern:
            return 1.0
        else:
            # Simple pattern similarity (could be enhanced with edit distance)
            baseline_steps = baseline_pattern.split("|") if baseline_pattern else []
            current_steps = current_pattern.split("|") if current_pattern else []
            
            if not baseline_steps and not current_steps:
                return 1.0
            
            # Jaccard similarity
            set1 = set(baseline_steps)
            set2 = set(current_steps)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _compare_decision_patterns(baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Compare decision-making patterns."""
        baseline_calls = baseline.get("llm_call_count", 0)
        current_calls = current.get("llm_call_count", 0)
        
        # Call count similarity
        if baseline_calls == 0 and current_calls == 0:
            call_score = 1.0
        elif baseline_calls == 0:
            call_score = 0.0
        else:
            call_score = max(0, 1 - abs(baseline_calls - current_calls) / baseline_calls)
        
        # Response length consistency
        baseline_length_consistency = baseline.get("response_length_consistency", 1.0)
        current_length_consistency = current.get("response_length_consistency", 1.0)
        length_score = 1 - abs(baseline_length_consistency - current_length_consistency)
        
        return (call_score + length_score) / 2
    
    @staticmethod
    def _compare_error_patterns(baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Compare error patterns."""
        baseline_rate = baseline.get("error_rate", 0)
        current_rate = current.get("error_rate", 0)
        
        # Lower error rates are better, but consistency is key
        rate_diff = abs(baseline_rate - current_rate)
        
        # If both have low error rates, that's good
        if baseline_rate <= 0.1 and current_rate <= 0.1:
            return 1.0 - rate_diff
        
        # If baseline had errors but current doesn't, that's an improvement
        if baseline_rate > current_rate:
            return min(1.0, 1.0 - rate_diff + 0.2)  # Bonus for improvement
        
        # If current has more errors, that's concerning
        return max(0.0, 1.0 - rate_diff * 2)


class TestFailure:
    """Represents a deterministic replay test failure."""
    
    def __init__(
        self,
        input_data: Any,
        consistency_score: float,
        expected_behavior: Dict[str, Any],
        actual_behavior: Dict[str, Any],
        threshold: float,
    ):
        self.input_data = input_data
        self.consistency_score = consistency_score
        self.expected_behavior = expected_behavior
        self.actual_behavior = actual_behavior
        self.threshold = threshold
        self.failure_id = generate_id()
        self.timestamp = get_current_time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure to dictionary format."""
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp,
            "input_data": str(self.input_data)[:500],  # Truncate for readability
            "consistency_score": self.consistency_score,
            "threshold": self.threshold,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
        }


class DeterministicReplayer:
    """Main class for deterministic replay testing of non-deterministic agents."""
    
    def __init__(
        self,
        consistency_threshold: float = 0.8,
        baseline_runs: int = 5,
        baseline_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the deterministic replayer.
        
        Args:
            consistency_threshold: Minimum consistency score to pass (0-1)
            baseline_runs: Number of runs to establish behavioral baseline
            baseline_dir: Directory to store baseline traces
        """
        self.consistency_threshold = consistency_threshold
        self.baseline_runs = baseline_runs
        self.baseline_dir = Path(baseline_dir) if baseline_dir else Path("baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines: Dict[str, BehavioralSignature] = {}
    
    def establish_baseline(
        self,
        agent_func: Callable,
        test_inputs: List[Any],
        baseline_name: str = "default",
    ) -> None:
        """Establish behavioral baseline for an agent function.
        
        Args:
            agent_func: The agent function to test
            test_inputs: List of test inputs to run
            baseline_name: Name for this baseline
        """
        print(f"🔄 Establishing baseline '{baseline_name}' with {len(test_inputs)} inputs...")
        
        all_traces = []
        
        for input_data in test_inputs:
            input_traces = []
            
            for run in range(self.baseline_runs):
                print(f"  📊 Input {test_inputs.index(input_data) + 1}/{len(test_inputs)}, Run {run + 1}/{self.baseline_runs}")
                
                # Create unique trace file for this run
                trace_file = self.baseline_dir / f"{baseline_name}_input_{test_inputs.index(input_data)}_run_{run}.json"
                
                with Trace(output=trace_file) as trace:
                    trace.metadata["baseline_name"] = baseline_name
                    trace.metadata["input_index"] = test_inputs.index(input_data)
                    trace.metadata["run_number"] = run
                    trace.metadata["input_data"] = str(input_data)[:200]  # Truncate for storage
                    
                    try:
                        result = agent_func(input_data)
                        trace.metadata["result"] = str(result)[:200]  # Truncate for storage
                    except Exception as e:
                        print(f"    ⚠️  Error in run {run + 1}: {e}")
                        # Exception is automatically captured by Trace context manager
                
                # Load the saved trace
                trace_data = load_trace(trace_file)
                input_traces.append(trace_data)
            
            all_traces.extend(input_traces)
        
        # Create behavioral signature from all traces
        signature = BehavioralSignature(all_traces)
        self.baselines[baseline_name] = signature
        
        # Save baseline metadata
        baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
        baseline_data = {
            "baseline_name": baseline_name,
            "created_at": get_current_time(),
            "test_inputs": [str(inp)[:100] for inp in test_inputs],
            "baseline_runs": self.baseline_runs,
            "consistency_threshold": self.consistency_threshold,
            "signature": signature.signature,
        }
        
        # Save baseline metadata without trace validation (different schema)
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Baseline '{baseline_name}' established successfully!")
        print(f"   📈 Analyzed {len(all_traces)} traces")
        print(f"   📁 Saved to {baseline_file}")
    
    def test_consistency(
        self,
        agent_func: Callable,
        test_inputs: List[Any],
        baseline_name: str = "default",
    ) -> List[TestFailure]:
        """Test agent consistency against established baseline.
        
        Args:
            agent_func: The agent function to test
            test_inputs: List of test inputs to run
            baseline_name: Name of baseline to compare against
            
        Returns:
            List of test failures (empty if all tests pass)
        """
        if baseline_name not in self.baselines:
            # Try to load baseline from disk
            baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
            if baseline_file.exists():
                with open(baseline_file, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
                # Reconstruct signature (simplified - in production you'd store/load full signature)
                self.baselines[baseline_name] = BehavioralSignature([])
                self.baselines[baseline_name].signature = baseline_data["signature"]
            else:
                raise ValueError(f"Baseline '{baseline_name}' not found. Run establish_baseline() first.")
        
        print(f"🧪 Testing consistency against baseline '{baseline_name}'...")
        
        baseline_signature = self.baselines[baseline_name]
        failures = []
        
        for i, input_data in enumerate(test_inputs):
            print(f"  🔍 Testing input {i + 1}/{len(test_inputs)}")
            
            # Run current version once
            trace_file = self.baseline_dir / f"test_{baseline_name}_input_{i}.json"
            
            with Trace(output=trace_file) as trace:
                trace.metadata["test_type"] = "consistency_test"
                trace.metadata["baseline_name"] = baseline_name
                trace.metadata["input_data"] = str(input_data)[:200]
                
                try:
                    result = agent_func(input_data)
                    trace.metadata["result"] = str(result)[:200]
                except Exception as e:
                    print(f"    ⚠️  Error in test: {e}")
            
            # Load and analyze the trace
            trace_data = load_trace(trace_file)
            current_signature = BehavioralSignature([trace_data])
            
            # Calculate consistency score
            consistency_score = ConsistencyScore.calculate(baseline_signature, current_signature)
            
            print(f"    📊 Consistency score: {consistency_score:.3f}")
            
            if consistency_score < self.consistency_threshold:
                failure = TestFailure(
                    input_data=input_data,
                    consistency_score=consistency_score,
                    expected_behavior=baseline_signature.signature,
                    actual_behavior=current_signature.signature,
                    threshold=self.consistency_threshold,
                )
                failures.append(failure)
                print(f"    ❌ FAIL: Score {consistency_score:.3f} < threshold {self.consistency_threshold}")
            else:
                print(f"    ✅ PASS: Score {consistency_score:.3f} >= threshold {self.consistency_threshold}")
        
        if failures:
            print(f"\n❌ {len(failures)}/{len(test_inputs)} tests failed")
            
            # Save failure report
            failure_report = {
                "test_timestamp": get_current_time(),
                "baseline_name": baseline_name,
                "threshold": self.consistency_threshold,
                "total_tests": len(test_inputs),
                "failed_tests": len(failures),
                "failures": [f.to_dict() for f in failures],
            }
            
            report_file = self.baseline_dir / f"failure_report_{baseline_name}_{get_current_time()[:19].replace(':', '-')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(failure_report, f, indent=2, ensure_ascii=False)
            print(f"📄 Failure report saved to {report_file}")
        else:
            print(f"\n✅ All {len(test_inputs)} tests passed!")
        
        return failures


def deterministic_replay(
    consistency_threshold: float = 0.8,
    baseline_runs: int = 5,
    baseline_name: str = "default",
    baseline_dir: Optional[Union[str, Path]] = None,
    auto_baseline: bool = False,
) -> Callable[[F], F]:
    """Decorator for deterministic replay testing of non-deterministic agents.
    
    Args:
        consistency_threshold: Minimum consistency score to pass (0-1)
        baseline_runs: Number of runs to establish behavioral baseline
        baseline_name: Name for the baseline
        baseline_dir: Directory to store baseline traces
        auto_baseline: Whether to automatically establish baseline on first run
        
    Returns:
        Decorated function with deterministic replay testing capabilities
    """
    def decorator(func: F) -> F:
        replayer = DeterministicReplayer(
            consistency_threshold=consistency_threshold,
            baseline_runs=baseline_runs,
            baseline_dir=baseline_dir,
        )
        
        # Store replayer on function for external access
        func._deterministic_replayer = replayer  # type: ignore
        func._baseline_name = baseline_name  # type: ignore
        func._auto_baseline = auto_baseline  # type: ignore
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Just run the function normally - testing is done via replayer methods
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator 