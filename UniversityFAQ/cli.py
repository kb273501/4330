"""Command-line evaluation for the University FAQ engine.

Reproduces the numbers in the report without starting Streamlit:

    python cli.py                                  # every retriever, no routing
    python cli.py --methods TF-IDF BM25 Dense Hybrid
    python cli.py --routers none rule trained --csv /tmp/results.csv
    python cli.py --ask "How do I renew a borrowed book?"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import faq_engine as fe
from intent import RuleIntent, TrainedIntent

DATA_DIR = Path(__file__).resolve().parent / "data"


def build_routers(names: list[str], data: list[dict]) -> list[tuple[str, object]]:
    mapping: dict[str, object] = {"none": None, "rule": RuleIntent()}
    if "trained" in names:
        try:
            mapping["trained"] = TrainedIntent().fit(data)
        except ValueError as error:
            print(f"trained router unavailable: {error}")
    return [(name, mapping[name]) for name in names if name in mapping]


def print_summary(summary: list[dict]) -> None:
    columns = (
        ("router", 8), ("method", 7), ("Recall@1", 9), ("Recall@5", 9),
        ("MRR@5", 7), ("Answer rate", 8), ("Refusal acc.", 9), ("BLEU", 7),
        ("ROUGE-L", 7), ("Task success", 8), ("Latency (ms)", 8),
    )
    header = "".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in summary:
        line = ""
        for name, width in columns:
            value = row.get(name, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            line += str(value).ljust(width)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the FAQ engine.")
    parser.add_argument("--faq", default=str(DATA_DIR / "faq.json"), help="knowledge base file")
    parser.add_argument("--tests", default=str(DATA_DIR / "tests.json"), help="test set file")
    parser.add_argument("--methods", nargs="+", default=["TF-IDF", "BM25"])
    parser.add_argument("--routers", nargs="+", default=["none"],
                        choices=("none", "rule", "trained"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=fe.GuardPolicy().min_coverage)
    parser.add_argument("--csv", help="write per-query rows here")
    parser.add_argument("--ask", help="answer one question and exit")
    args = parser.parse_args()

    data = fe.parse_faq(Path(args.faq).read_bytes(), Path(args.faq).name)
    engine = fe.Engine(data)
    policy = fe.GuardPolicy(min_coverage=args.coverage)

    if args.ask:
        hits, verdict, latency = engine.retrieve(args.ask, args.methods[0], args.k, policy)
        print(f"query: {args.ask}")
        print(f"accepted: {verdict.accepted}  coverage: {verdict.coverage:.2f}  {latency:.2f} ms")
        if verdict.accepted:
            print(f"answer: {fe.answer_from_evidence(hits[verdict.answer_index], fe.lang(args.ask))}")
            print(f"source: {hits[verdict.answer_index]['source']}")
        else:
            print(f"refusal: {fe.refusal_text(verdict, fe.lang(args.ask))}")
        return

    tests = fe.parse_tests(Path(args.tests).read_bytes(), Path(args.tests).name)
    routers = build_routers(args.routers, data)
    rows, summary = fe.evaluate(
        engine, tests, args.methods, k=args.k, policy=policy, routers=routers
    )
    print_summary(summary)
    print(f"\n{len(tests)} queries, coverage threshold {args.coverage:.2f}")

    if args.csv:
        fields = sorted({key for row in rows for key in row})
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore", restval=""
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-query rows written to {args.csv}")


if __name__ == "__main__":
    main()
