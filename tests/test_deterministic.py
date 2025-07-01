"""Tests for deterministic replay testing functionality."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agentcheck


class TestDeterministicReplay(unittest.TestCase):
    """Test cases for deterministic replay functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_dir = Path(self.temp_dir) / "baselines"
        
        # Simple test agent
        def simple_agent(x: int) -> str:
            with agentcheck.trace() as trace:
                trace.add_step("calculation", {"input": x})
                result = f"The answer is {x * 2}"
                trace.add_step("response", {"output": result})
                return result
        
        self.simple_agent = simple_agent
    
    def test_behavioral_signature_creation(self):
        """Test creating behavioral signatures from traces."""
        # Create mock traces
        traces = [
            {
                "steps": [
                    {"type": "calculation", "input": {"value": 5}},
                    {"type": "response", "output": {"result": "10"}},
                ],
                "metadata": {}
            },
            {
                "steps": [
                    {"type": "calculation", "input": {"value": 3}},
                    {"type": "response", "output": {"result": "6"}},
                ],
                "metadata": {}
            }
        ]
        
        signature = agentcheck.BehavioralSignature(traces)
        
        # Test signature components
        self.assertIn("step_count", signature.signature)
        self.assertIn("step_types", signature.signature)
        self.assertEqual(signature.signature["step_count"]["mean"], 2.0)
        self.assertEqual(signature.signature["step_count"]["min"], 2)
        self.assertEqual(signature.signature["step_count"]["max"], 2)
    
    def test_consistency_score_calculation(self):
        """Test consistency score calculation between signatures."""
        # Create identical signatures
        traces1 = [
            {"steps": [{"type": "step1"}, {"type": "step2"}], "metadata": {}}
        ]
        traces2 = [
            {"steps": [{"type": "step1"}, {"type": "step2"}], "metadata": {}}
        ]
        
        sig1 = agentcheck.BehavioralSignature(traces1)
        sig2 = agentcheck.BehavioralSignature(traces2)
        
        score = agentcheck.ConsistencyScore.calculate(sig1, sig2)
        
        # Identical signatures should have high consistency
        self.assertGreater(score, 0.8)
    
    def test_deterministic_replayer_initialization(self):
        """Test DeterministicReplayer initialization."""
        replayer = agentcheck.DeterministicReplayer(
            consistency_threshold=0.9,
            baseline_runs=3,
            baseline_dir=self.baseline_dir
        )
        
        self.assertEqual(replayer.consistency_threshold, 0.9)
        self.assertEqual(replayer.baseline_runs, 3)
        self.assertEqual(replayer.baseline_dir, self.baseline_dir)
        self.assertTrue(self.baseline_dir.exists())
    
    def test_establish_baseline(self):
        """Test establishing a behavioral baseline."""
        replayer = agentcheck.DeterministicReplayer(
            baseline_runs=2,  # Reduced for testing
            baseline_dir=self.baseline_dir
        )
        
        test_inputs = [1, 2]
        
        # Mock the trace saving to avoid file I/O in tests
        with patch('agentcheck.deterministic.load_trace') as mock_load:
            mock_load.return_value = {
                "steps": [{"type": "test"}],
                "metadata": {},
                "trace_id": "test"
            }
            
            with patch('agentcheck.deterministic.save_trace'):
                replayer.establish_baseline(
                    agent_func=self.simple_agent,
                    test_inputs=test_inputs,
                    baseline_name="test_baseline"
                )
        
        # Check that baseline was created
        self.assertIn("test_baseline", replayer.baselines)
    
    def test_test_failure_creation(self):
        """Test TestFailure object creation."""
        failure = agentcheck.TestFailure(
            input_data="test input",
            consistency_score=0.5,
            expected_behavior={"pattern": "A"},
            actual_behavior={"pattern": "B"},
            threshold=0.8
        )
        
        self.assertEqual(failure.input_data, "test input")
        self.assertEqual(failure.consistency_score, 0.5)
        self.assertEqual(failure.threshold, 0.8)
        self.assertIsNotNone(failure.failure_id)
        self.assertIsNotNone(failure.timestamp)
        
        # Test dictionary conversion
        failure_dict = failure.to_dict()
        self.assertIn("failure_id", failure_dict)
        self.assertIn("consistency_score", failure_dict)
        self.assertIn("threshold", failure_dict)
    
    def test_deterministic_replay_decorator(self):
        """Test the deterministic_replay decorator."""
        @agentcheck.deterministic_replay(
            consistency_threshold=0.7,
            baseline_runs=2,
            baseline_name="decorated_test",
            baseline_dir=self.baseline_dir
        )
        def decorated_agent(x: int) -> str:
            return f"Result: {x}"
        
        # Test that decorator adds required attributes
        self.assertTrue(hasattr(decorated_agent, '_deterministic_replayer'))
        self.assertTrue(hasattr(decorated_agent, '_baseline_name'))
        self.assertEqual(decorated_agent._baseline_name, "decorated_test")
        
        # Test that function still works normally
        result = decorated_agent(5)
        self.assertEqual(result, "Result: 5")
        
        # Test replayer configuration
        replayer = decorated_agent._deterministic_replayer
        self.assertEqual(replayer.consistency_threshold, 0.7)
        self.assertEqual(replayer.baseline_runs, 2)
    
    def test_step_count_comparison(self):
        """Test step count comparison in consistency scoring."""
        baseline = {"mean": 5.0}
        current = {"mean": 5.0}
        
        score = agentcheck.ConsistencyScore._compare_step_counts(baseline, current)
        self.assertEqual(score, 1.0)  # Perfect match
        
        # Test with difference
        current = {"mean": 6.0}
        score = agentcheck.ConsistencyScore._compare_step_counts(baseline, current)
        self.assertLess(score, 1.0)  # Should be less than perfect
        self.assertGreater(score, 0.0)  # But still positive
    
    def test_step_pattern_comparison(self):
        """Test step pattern comparison in consistency scoring."""
        baseline = {"common_pattern": "step1|step2|step3"}
        current = {"common_pattern": "step1|step2|step3"}
        
        score = agentcheck.ConsistencyScore._compare_step_patterns(baseline, current)
        self.assertEqual(score, 1.0)  # Perfect match
        
        # Test with different pattern
        current = {"common_pattern": "step1|step3"}
        score = agentcheck.ConsistencyScore._compare_step_patterns(baseline, current)
        self.assertLess(score, 1.0)  # Should be less than perfect
        self.assertGreater(score, 0.0)  # But still positive due to overlap
    
    def test_error_pattern_comparison(self):
        """Test error pattern comparison in consistency scoring."""
        baseline = {"error_rate": 0.0}
        current = {"error_rate": 0.0}
        
        score = agentcheck.ConsistencyScore._compare_error_patterns(baseline, current)
        self.assertEqual(score, 1.0)  # Perfect match - no errors
        
        # Test with increased error rate
        current = {"error_rate": 0.2}
        score = agentcheck.ConsistencyScore._compare_error_patterns(baseline, current)
        self.assertLess(score, 1.0)  # Should be penalized
    
    def test_decision_pattern_comparison(self):
        """Test decision pattern comparison in consistency scoring."""
        baseline = {
            "llm_call_count": 3,
            "response_length_consistency": 0.9
        }
        current = {
            "llm_call_count": 3,
            "response_length_consistency": 0.9
        }
        
        score = agentcheck.ConsistencyScore._compare_decision_patterns(baseline, current)
        self.assertEqual(score, 1.0)  # Perfect match
        
        # Test with different call count
        current = {
            "llm_call_count": 5,
            "response_length_consistency": 0.9
        }
        score = agentcheck.ConsistencyScore._compare_decision_patterns(baseline, current)
        self.assertLess(score, 1.0)  # Should be less than perfect
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main() 