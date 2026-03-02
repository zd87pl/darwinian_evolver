"""
Apply the best organism's changes back to the real repository.

Usage:
    python -m darwinian_evolver apply --output_dir /tmp/evolution_output
    python -m darwinian_evolver apply --output_dir /tmp/evolution_output --branch evolution/fix
    python -m darwinian_evolver apply --output_dir /tmp/evolution_output --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path


def load_best_organism(output_dir: Path) -> tuple[dict, dict, float]:
    """Load the best organism from results.jsonl.

    Returns (organism_dict, evaluation_result_dict, score).
    """
    results_file = output_dir / "results.jsonl"
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

    if not data:
        print("Error: No iteration data found in results.jsonl")
        sys.exit(1)

    # Find the best organism across all iterations
    best_organism = None
    best_result = None
    best_score = -1.0

    for iteration_data in data:
        population = iteration_data.get("population", {})
        organisms = population.get("organisms", [])
        for org_data in organisms:
            result = org_data.get("evaluation_result", {})
            score = result.get("score", 0)
            if score > best_score:
                best_score = score
                best_organism = org_data.get("organism", {})
                best_result = result

    if best_organism is None:
        print("Error: No organisms found in results")
        sys.exit(1)

    return best_organism, best_result, best_score


def compute_diff(repo_root: str, file_contents: dict[str, str]) -> str:
    """Compute a unified diff between repo files and organism file contents."""
    diff_parts = []
    for path in sorted(file_contents.keys()):
        new_content = file_contents[path]
        repo_file = os.path.join(repo_root, path)

        if os.path.exists(repo_file):
            with open(repo_file) as f:
                old_content = f.read()
        else:
            old_content = ""

        if old_content != new_content:
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
            diff_str = "".join(diff)
            if diff_str:
                diff_parts.append(diff_str)

    return "\n".join(diff_parts)


def apply_changes(
    repo_root: str,
    file_contents: dict[str, str],
    branch: str | None = None,
) -> None:
    """Write the organism's file contents to the repo."""
    if branch:
        # Create and checkout a new branch
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Created branch: {branch}")
        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                print(f"Branch {branch} already exists. Checking out...")
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                print(f"Error creating branch: {e.stderr}")
                sys.exit(1)

    files_written = 0
    files_unchanged = 0
    for path, content in sorted(file_contents.items()):
        full_path = os.path.join(repo_root, path)

        # Check if content actually differs
        if os.path.exists(full_path):
            with open(full_path) as f:
                if f.read() == content:
                    files_unchanged += 1
                    continue

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        files_written += 1
        print(f"  Updated: {path}")

    print(f"\nApplied: {files_written} file(s) modified, {files_unchanged} unchanged")


def write_patch(diff_text: str, output_path: Path) -> None:
    """Write the diff to a patch file."""
    output_path.write_text(diff_text)
    print(f"Patch written to: {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply best organism changes to the repository")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to evolution output directory")
    parser.add_argument("--repo_root", type=str, default=None, help="Override the repository root path")
    parser.add_argument(
        "--branch", type=str, default=None, help="Create a new git branch before applying (e.g. evolution/fix)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show diff without applying changes")
    parser.add_argument("--patch", type=Path, default=None, help="Write diff to a patch file")
    parser.add_argument("--iteration", type=int, default=None, help="Apply from a specific iteration (default: best across all)")

    args = parser.parse_args(argv)

    organism, result, score = load_best_organism(args.output_dir)

    file_contents = organism.get("file_contents", {})
    repo_root = args.repo_root or organism.get("repo_root", "")

    if not repo_root:
        print("Error: Could not determine repo_root. Use --repo_root to specify it.")
        sys.exit(1)

    if not os.path.isdir(repo_root):
        print(f"Error: Repository root not found: {repo_root}")
        sys.exit(1)

    # Summary
    print(f"Best organism score: {score:.4f}")
    num_passed = result.get("num_passed")
    num_total = result.get("num_total")
    if num_passed is not None:
        print(f"Tests: {num_passed}/{num_total} passed")
    if organism.get("from_change_summary"):
        print(f"Summary: {organism['from_change_summary'][:200]}")
    print(f"Files: {len(file_contents)}")
    print()

    # Compute diff
    diff_text = compute_diff(repo_root, file_contents)

    if not diff_text:
        print("No changes to apply — the organism matches the current repo state.")
        return

    if args.patch:
        write_patch(diff_text, args.patch)

    if args.dry_run:
        print("--- Diff (dry run) ---")
        print(diff_text)
        return

    # Apply
    apply_changes(repo_root, file_contents, branch=args.branch)


if __name__ == "__main__":
    main()
