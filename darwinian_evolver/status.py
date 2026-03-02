"""
Inspect evolution run results from the command line.

Usage:
    python -m darwinian_evolver status --output_dir /tmp/evolution_output
    python -m darwinian_evolver status --output_dir /tmp/evolution_output --diff
    python -m darwinian_evolver status --output_dir /tmp/evolution_output --files
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path


def load_results(results_file: Path) -> list[dict]:
    """Load all iteration data from results.jsonl."""
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)

    data = []
    for line in results_file.read_text().strip().split("\n"):
        if line.strip():
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def get_best_from_iteration(iteration_data: dict) -> tuple[dict, dict, float] | None:
    """Get (organism, result, score) for the best organism in an iteration."""
    population = iteration_data.get("population", {})
    organisms = population.get("organisms", [])
    if not organisms:
        return None

    best = max(organisms, key=lambda o: o.get("evaluation_result", {}).get("score", 0))
    result = best.get("evaluation_result", {})
    return best.get("organism", {}), result, result.get("score", 0)


def render_status(data: list[dict]) -> None:
    """Print a summary of the evolution run."""
    if not data:
        print("No iteration data found.")
        return

    print(f"Iterations completed: {len(data)}")
    print()

    # Score progression
    print("Score progression:")
    best_overall_score = -1.0
    best_overall_iteration = 0

    for d in data:
        iteration = d["iteration"]
        best = get_best_from_iteration(d)
        if best is None:
            print(f"  Iteration {iteration}: (no organisms)")
            continue

        _, result, score = best
        pop_size = len(d.get("population", {}).get("organisms", []))

        # Build details
        details = []
        num_passed = result.get("num_passed")
        num_total = result.get("num_total")
        if num_passed is not None:
            details.append(f"tests: {num_passed}/{num_total}")

        criteria = result.get("criteria_scores", {})
        if criteria:
            crit_str = ", ".join(f"{k}: {v:.0f}/10" for k, v in criteria.items())
            details.append(crit_str)

        detail_str = f" ({', '.join(details)})" if details else ""

        marker = ""
        if score > best_overall_score:
            best_overall_score = score
            best_overall_iteration = iteration
            marker = " *"

        print(f"  Iteration {iteration}: {score:.4f} (pop: {pop_size}){detail_str}{marker}")

    print()
    print(f"Best score: {best_overall_score:.4f} (iteration {best_overall_iteration})")

    # Latest organism summary
    latest_best = get_best_from_iteration(data[-1])
    if latest_best:
        organism, result, score = latest_best
        summary = organism.get("from_change_summary", "")
        if summary:
            print(f"\nLatest best organism summary:\n  {summary[:300]}")

    # Verification failures
    latest = data[-1]
    failures = latest.get("verification_failures", [])
    if failures:
        print(f"\nVerification failures in latest iteration: {len(failures)}")


def render_diff(data: list[dict]) -> None:
    """Show the diff between the initial organism and the current best."""
    if not data:
        print("No iteration data found.")
        return

    # Get first iteration's best (often the initial organism)
    first_best = get_best_from_iteration(data[0])
    latest_best = get_best_from_iteration(data[-1])

    if not first_best or not latest_best:
        print("Could not find organisms to compare.")
        return

    first_files = first_best[0].get("file_contents", {})
    latest_files = latest_best[0].get("file_contents", {})

    all_files = sorted(set(first_files.keys()) | set(latest_files.keys()))
    has_changes = False

    for path in all_files:
        old_content = first_files.get(path, "")
        new_content = latest_files.get(path, "")

        if old_content == new_content:
            continue

        has_changes = True
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"initial/{path}",
            tofile=f"evolved/{path}",
        )
        print("".join(diff))

    if not has_changes:
        print("No changes between initial and current best organism.")


def render_files(data: list[dict]) -> None:
    """List the files in the best organism."""
    if not data:
        print("No iteration data found.")
        return

    latest_best = get_best_from_iteration(data[-1])
    if not latest_best:
        print("No organisms found.")
        return

    organism = latest_best[0]
    file_contents = organism.get("file_contents", {})

    print(f"Files in best organism ({len(file_contents)}):")
    for path in sorted(file_contents.keys()):
        content = file_contents[path]
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        print(f"  {path} ({line_count} lines)")


def render_mutations(data: list[dict]) -> None:
    """Show mutation log across iterations."""
    if not data:
        print("No iteration data found.")
        return

    for d in data:
        iteration = d["iteration"]
        organisms = d.get("population", {}).get("organisms", [])
        for org_data in organisms:
            organism = org_data.get("organism", {})
            result = org_data.get("evaluation_result", {})
            summary = organism.get("from_change_summary")
            if summary:
                score = result.get("score", 0)
                org_id = organism.get("id", "?")[:8]
                parent_id = organism.get("parent_id", "root")
                if parent_id and parent_id != "root":
                    parent_id = parent_id[:8]
                print(f"  [{iteration}] {score:.2f} {parent_id} -> {org_id}: {summary[:120]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect evolution run results")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to evolution output directory")
    parser.add_argument("--diff", action="store_true", help="Show diff between initial and best organism")
    parser.add_argument("--files", action="store_true", help="List files in the best organism")
    parser.add_argument("--mutations", action="store_true", help="Show mutation log")

    args = parser.parse_args(argv)

    results_file = args.output_dir / "results.jsonl"
    data = load_results(results_file)

    if args.diff:
        render_diff(data)
    elif args.files:
        render_files(data)
    elif args.mutations:
        render_mutations(data)
    else:
        render_status(data)


if __name__ == "__main__":
    main()
