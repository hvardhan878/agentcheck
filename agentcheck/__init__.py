"""agentcheck: Trace ⋅ Replay ⋅ Test your AI agents like real software."""

from .trace import Trace, trace
from .replay import replay_trace
from .diff import diff_traces
from .asserts import assert_trace
from .deterministic import (
    DeterministicReplayer,
    BehavioralSignature,
    ConsistencyScore,
    TestFailure,
    deterministic_replay,
)

__version__ = "0.1.0"
__all__ = [
    "Trace", 
    "trace", 
    "replay_trace", 
    "diff_traces", 
    "assert_trace",
    "DeterministicReplayer",
    "BehavioralSignature", 
    "ConsistencyScore",
    "TestFailure",
    "deterministic_replay",
] 