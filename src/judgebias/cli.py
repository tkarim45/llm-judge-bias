"""Command line for the judge-bias benchmark.

    llm-judge-bias                     # real Claude judge on Bedrock, all pairs in both orders
    llm-judge-bias --backend anthropic
    llm-judge-bias --offline           # deterministic biased fake judge, no keys
    llm-judge-bias --json results.json

A real run makes two short judge calls per pair (each pair judged in both slot orders).
"""
from __future__ import annotations

import argparse
import json
import sys

from .benchmark import BiasFakeJudge, format_report, run
from .data import all_pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure position/verbosity/assertiveness bias in an LLM judge.")
    ap.add_argument("--backend", default="bedrock", choices=["bedrock", "anthropic"])
    ap.add_argument("--offline", action="store_true", help="deterministic biased fake judge, no keys")
    ap.add_argument("--fake-mode", default="prefer_longer", choices=["prefer_longer", "always_first"])
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    args = ap.parse_args(argv)

    if args.offline:
        client = BiasFakeJudge(mode=args.fake_mode)
    else:
        client = __import__("judgebias.llm", fromlist=["build_client"]).build_client(args.backend)
    result = run(client, all_pairs())

    print(format_report(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
