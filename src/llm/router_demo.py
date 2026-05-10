"""Combined demo for rule-based and model-based thinking routers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .lmtrain import DEMO_QUERIES, OUTPUT_DIR, THRESHOLD, ModelBasedRouter
from .rule_based_router import explain_route, rule_statuses


def _mode(use_thinking: bool) -> str:
    return "THINKING" if use_thinking else "standard"


def format_rule_router_demo(
    queries: tuple[str, ...] = DEMO_QUERIES,
    use_secondary_rules: bool = False,
) -> str:
    """Format a detailed rule-router demo with every rule result."""
    lines = [
        "Running rule-based router",
        "=" * 25,
        f"Secondary rules: {use_secondary_rules}",
        "",
    ]
    for query in queries:
        decision = explain_route(query, use_secondary_rules=use_secondary_rules)
        lines.append(f"Query: {query}")
        lines.append("Rule checks:")
        for status in rule_statuses(query, use_secondary_rules=use_secondary_rules):
            lines.append(f"  - {status.name:<15} {status.fired}")
        lines.append(f"Signals fired: {', '.join(decision.signals) or '-'}")
        lines.append(f"Decision: {_mode(decision.use_thinking)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_model_router_demo(
    router: ModelBasedRouter,
    queries: tuple[str, ...] = DEMO_QUERIES,
) -> str:
    """Format model-router demo output with score, threshold, and mode."""
    lines = [
        "Running model-based router",
        "=" * 26,
        f"Threshold: {router.threshold:.2f}",
        "",
        f"{'Query':<38} {'P(thinking)':>11} {'Mode':>10}",
        "-" * 63,
    ]
    for query in queries:
        decision = router.route(query)
        lines.append(
            f"{decision.query:<38} {decision.probability:>11.4f} "
            f"{_mode(decision.use_thinking):>10}"
        )
    return "\n".join(lines)


def format_combined_demo(
    router: ModelBasedRouter,
    queries: tuple[str, ...] = DEMO_QUERIES,
    use_secondary_rules: bool = False,
) -> str:
    """Format both router demos in one output."""
    return "\n\n".join(
        [
            format_rule_router_demo(
                queries=queries,
                use_secondary_rules=use_secondary_rules,
            ),
            format_model_router_demo(router=router, queries=queries),
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run both thinking-router demos with detailed output.",
    )
    parser.add_argument(
        "-demo",
        "--demo",
        action="store_true",
        help="Run the combined router demo. This is the default behavior.",
    )
    parser.add_argument(
        "--model-path",
        default=str(OUTPUT_DIR),
        help=f"Path to the trained model-router directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Model-router threshold (default: {THRESHOLD}).",
    )
    parser.add_argument(
        "--secondary",
        action="store_true",
        help="Enable secondary rules in the rule-based router.",
    )
    return parser


def main() -> None:
    """Run the combined router demo."""
    args = _build_parser().parse_args()
    router = ModelBasedRouter.load(
        model_path=Path(args.model_path),
        threshold=args.threshold,
    )
    print(format_combined_demo(router=router, use_secondary_rules=args.secondary))


if __name__ == "__main__":
    main()
