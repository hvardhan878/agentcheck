"""Command-line interface for agentcheck."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from . import __version__
from .asserts import assert_trace
from .diff import diff_traces
from .replay import replay_trace
from .utils import load_trace, pretty_print_json

console = Console()


def cmd_trace(args: argparse.Namespace) -> None:
    """Handle the trace command."""
    import subprocess
    
    # For trace command, we run the specified Python script/command
    # The script should be instrumented with @agentcheck.trace decorator
    
    if not args.command:
        console.print("❌ No command specified to trace", style="red")
        sys.exit(1)
    
    try:
        # Run the command
        result = subprocess.run(args.command, check=True, shell=True)
        
        if args.output and Path(args.output).exists():
            console.print(f"✅ Trace saved to: {args.output}", style="green")
        else:
            console.print("⚠️  No trace file generated. Make sure your code uses @agentcheck.trace decorator", style="yellow")
            
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Command failed with exit code {e.returncode}", style="red")
        sys.exit(e.returncode)


def cmd_replay(args: argparse.Namespace) -> None:
    """Handle the replay command."""
    try:
        result = replay_trace(
            trace_file=args.trace_file,
            output_file=args.output,
            model_override=args.model,
        )
        console.print(f"✅ Replay completed successfully", style="green")
        
        if args.show:
            pretty_print_json(result)
            
    except Exception as e:
        console.print(f"❌ Replay failed: {e}", style="red")
        sys.exit(1)


def cmd_diff(args: argparse.Namespace) -> None:
    """Handle the diff command."""
    try:
        result = diff_traces(
            trace_a_file=args.trace_a,
            trace_b_file=args.trace_b,
            output_file=args.output,
        )
        
        # The diff is already displayed by the diff_traces function
        console.print("✅ Diff completed successfully", style="green")
        
    except Exception as e:
        console.print(f"❌ Diff failed: {e}", style="red")
        sys.exit(1)


def cmd_assert(args: argparse.Namespace) -> None:
    """Handle the assert command."""
    try:
        assert_trace(
            trace_file=args.trace_file,
            contains=args.contains,
            not_contains=args.not_contains,
            jsonpath=args.jsonpath,
            regex=args.regex,
            min_cost=args.min_cost,
            max_cost=args.max_cost,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            exit_on_failure=True,  # CLI always exits on failure
        )
        
    except Exception as e:
        console.print(f"❌ Assertion failed: {e}", style="red")
        sys.exit(1)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle the show command."""
    try:
        trace_data = load_trace(args.trace_file)
        pretty_print_json(trace_data)
        
    except Exception as e:
        console.print(f"❌ Failed to show trace: {e}", style="red")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentcheck",
        description="Trace ⋅ Replay ⋅ Test your AI agents like real software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"agentcheck {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Trace command
    trace_parser = subparsers.add_parser(
        "trace",
        help="Run a command and trace its execution",
    )
    trace_parser.add_argument(
        "command",
        nargs="?",
        help="Command to run and trace",
    )
    trace_parser.add_argument(
        "--output", "-o",
        help="Output trace file path",
    )
    trace_parser.set_defaults(func=cmd_trace)
    
    # Replay command
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a trace file",
    )
    replay_parser.add_argument(
        "trace_file",
        help="Path to the trace file to replay",
    )
    replay_parser.add_argument(
        "--output", "-o",
        help="Output file for the new trace",
    )
    replay_parser.add_argument(
        "--model", "-m",
        help="Override model for LLM calls",
    )
    replay_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the replay results",
    )
    replay_parser.set_defaults(func=cmd_replay)
    
    # Diff command
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two trace files",
    )
    diff_parser.add_argument(
        "trace_a",
        help="Path to the first trace file (baseline)",
    )
    diff_parser.add_argument(
        "trace_b", 
        help="Path to the second trace file (comparison)",
    )
    diff_parser.add_argument(
        "--output", "-o",
        help="Output file to save diff results",
    )
    diff_parser.set_defaults(func=cmd_diff)
    
    # Assert command
    assert_parser = subparsers.add_parser(
        "assert",
        help="Make assertions about a trace file",
    )
    assert_parser.add_argument(
        "trace_file",
        help="Path to the trace file",
    )
    assert_parser.add_argument(
        "--contains",
        help="Assert that trace contains this string",
    )
    assert_parser.add_argument(
        "--not-contains",
        dest="not_contains",
        help="Assert that trace does NOT contain this string",
    )
    assert_parser.add_argument(
        "--jsonpath",
        help="JSONPath expression to extract data before checking",
    )
    assert_parser.add_argument(
        "--regex",
        help="Regex pattern to match against content",
    )
    assert_parser.add_argument(
        "--min-cost",
        type=float,
        help="Minimum expected cost",
    )
    assert_parser.add_argument(
        "--max-cost", 
        type=float,
        help="Maximum expected cost",
    )
    assert_parser.add_argument(
        "--min-steps",
        type=int,
        help="Minimum number of steps",
    )
    assert_parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps",
    )
    assert_parser.set_defaults(func=cmd_assert)
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Display a trace file with pretty formatting",
    )
    show_parser.add_argument(
        "trace_file",
        help="Path to the trace file to display",
    )
    show_parser.set_defaults(func=cmd_show)
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Call the appropriate command function
    args.func(args)


if __name__ == "__main__":
    main() 