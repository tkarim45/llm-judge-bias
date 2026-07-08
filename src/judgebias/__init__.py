"""llm-judge-bias — measure position, verbosity, and assertiveness bias in an LLM-as-judge on
controlled answer pairs, by judging each pair in both orders and isolating what the verdict actually
depends on."""
from .benchmark import format_report, parse_choice, run

__all__ = ["run", "format_report", "parse_choice"]
