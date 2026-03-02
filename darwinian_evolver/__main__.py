import argparse
import json
import sys
import tempfile
from pathlib import Path

from darwinian_evolver.cli_common import build_hyperparameter_config_from_args
from darwinian_evolver.cli_common import parse_learning_log_view_type
from darwinian_evolver.cli_common import register_hyperparameter_args
from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
from darwinian_evolver.evolve_problem_loop import IterationSnapshot
from darwinian_evolver.problems.registry import AVAILABLE_PROBLEMS
from darwinian_evolver.problems.registry import CONFIGURABLE_PROBLEMS
from darwinian_evolver.problems.registry import register_repo_task
from darwinian_evolver.problems.registry import register_spec_task
from darwinian_evolver.storage import upload_bytes_to_s3
from darwinian_evolver.storage import upload_file_to_s3_fixed_path

# Register configurable problems
register_repo_task()
register_spec_task()

ALL_PROBLEM_NAMES = list(AVAILABLE_PROBLEMS.keys()) + list(CONFIGURABLE_PROBLEMS.keys())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Darwin Prompt evolution loop."""
    arg_parser = argparse.ArgumentParser(description="Run the Darwin Prompt evolution loop on a given problem.")
    arg_parser.add_argument(
        "problem",
        type=str,
        choices=ALL_PROBLEM_NAMES,
        help="The problem to evolve. Available problems: " + ", ".join(ALL_PROBLEM_NAMES),
    )

    hyperparameter_args = arg_parser.add_argument_group("Hyperparameters")
    register_hyperparameter_args(hyperparameter_args)

    runtime_args = arg_parser.add_argument_group("Runtime")
    runtime_args.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        required=False,
        help="The number of iterations to run the evolution loop for. Default is 5.",
    )
    runtime_args.add_argument(
        "--mutator_concurrency",
        type=int,
        default=10,
        required=False,
        help="The maximum number of mutators to run concurrently. Default is 10.",
    )
    runtime_args.add_argument(
        "--evaluator_concurrency",
        type=int,
        default=10,
        required=False,
        help="The maximum number of evaluators to run concurrently. Default is 10.",
    )
    runtime_args.add_argument(
        "--use_process_pool",
        action="store_true",
        default=False,
        help="Use multi-processing instead of multi-threading to run mutators and evaluators.",
    )
    runtime_args.add_argument(
        "--early_stop_score",
        type=float,
        default=None,
        required=False,
        help="Stop evolution early when best score reaches this threshold (e.g. 1.0 for all tests passing).",
    )

    fixed_tree_mode_args = arg_parser.add_argument_group("Fixed Tree Mode")
    fixed_tree_mode_args.add_argument(
        "--fixed_children_per_generation",
        type=int,
        nargs="+",
        required=False,
        help="Space-separated list of number of children per generation for fixed tree mode (e.g., '10 1' or '6 4 3 2 2'). When provided, enables fixed tree structure instead of weighted parent sampling. Each iteration will produce a generation. Iterations beyond the list length will cycle through the list again.",
    )

    input_output_args = arg_parser.add_argument_group("Input/Output")
    input_output_args.add_argument(
        "--resume_from_snapshot",
        type=Path,
        required=False,
        help="Path to the snapshot file to resume from.",
    )
    input_output_args.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        help="Output directory for logs and evaluator results. Will be created if it does not exist.",
    )
    input_output_args.add_argument(
        "--overwrite_snapshots",
        action="store_true",
        help="Overwrite existing snapshot files.",
    )
    input_output_args.add_argument(
        "--s3_dir",
        type=Path,
        required=False,
        help="S3 path to upload results to, relative to the s3://int8-shared-internal/ bucket.",
    )

    # repo_task-specific arguments
    repo_task_args = arg_parser.add_argument_group("repo_task options (only used when problem=repo_task)")
    repo_task_args.add_argument(
        "--repo_root",
        type=str,
        required=False,
        help="Path to the target git repository.",
    )
    repo_task_args.add_argument(
        "--task",
        type=str,
        required=False,
        help="Natural language description of the coding task.",
    )
    repo_task_args.add_argument(
        "--test_command",
        type=str,
        required=False,
        help='Command to run tests, e.g. "pytest --tb=short".',
    )
    repo_task_args.add_argument(
        "--setup_command",
        type=str,
        required=False,
        help="Optional setup command to run before tests (e.g. pip install -e .).",
    )
    repo_task_args.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=False,
        help="Explicit list of files to evolve. Auto-detected if omitted.",
    )
    repo_task_args.add_argument(
        "--agent_model",
        type=str,
        default="claude-sonnet-4-20250514",
        required=False,
        help="Model for the agentic mutator. Default: claude-sonnet-4-20250514.",
    )
    repo_task_args.add_argument(
        "--agent_max_turns",
        type=int,
        default=25,
        required=False,
        help="Max tool-use turns per mutation. Default: 25.",
    )
    repo_task_args.add_argument(
        "--test_timeout",
        type=int,
        default=300,
        required=False,
        help="Timeout in seconds for the test command. Default: 300.",
    )

    # spec_task-specific arguments
    spec_task_args = arg_parser.add_argument_group("spec_task options (only used when problem=spec_task)")
    spec_task_args.add_argument(
        "--spec",
        type=str,
        required=False,
        help="The specification that the code should implement (natural language).",
    )
    spec_task_args.add_argument(
        "--spec_file",
        type=Path,
        required=False,
        help="Path to a file containing the specification (alternative to --spec).",
    )
    spec_task_args.add_argument(
        "--validation_command",
        type=str,
        required=False,
        help='Optional validation command (e.g. "python -c \'import mymodule\'", "ruff check .").',
    )
    spec_task_args.add_argument(
        "--judge_model",
        type=str,
        default="claude-sonnet-4-20250514",
        required=False,
        help="Model for the LLM judge evaluator. Default: claude-sonnet-4-20250514.",
    )

    return arg_parser.parse_args()


def print_snapshot_summary(snapshot: IterationSnapshot) -> None:
    print(f"Iteration {snapshot.iteration}:")
    print("  Best score:", snapshot.best_organism_result[1].score)
    print(
        "  Best organism:",
        snapshot.best_organism_result[0].id,
        str(
            snapshot.best_organism_result[0].model_dump(
                exclude={
                    # Exclude all the fields from the Organism base class to highlight the problem-specific fields
                    "parent",
                    "additional_parents",
                    "id",
                    "from_failure_cases",
                    "from_learning_log_entries",
                    "from_change_summary",
                    "visualizer_props",
                }
            )
        )[:80]
        + "...",
    )
    print("  Population size:", snapshot.population_size)
    print("  Evolver stats: ")
    for s, v in snapshot.evolver_stats.model_dump().items():
        print(f"    {s}: {v}")


def _print_cost_summary(problem, final: bool = False) -> None:
    """Print token usage from mutators and evaluators."""
    total_input = 0
    total_output = 0
    total_calls = 0

    for mutator in problem.mutators:
        if hasattr(mutator, "total_input_tokens"):
            total_input += mutator.total_input_tokens
            total_output += mutator.total_output_tokens
            total_calls += mutator.total_api_calls

    evaluator = problem.evaluator
    if hasattr(evaluator, "total_input_tokens"):
        total_input += evaluator.total_input_tokens
        total_output += evaluator.total_output_tokens
        total_calls += evaluator.total_api_calls

    if total_calls == 0:
        return

    total_tokens = total_input + total_output
    # Approximate cost (Sonnet pricing: $3/MTok input, $15/MTok output)
    approx_cost = (total_input * 3.0 + total_output * 15.0) / 1_000_000

    if final:
        print(f"\n  Total API calls: {total_calls}")
        print(f"  Total tokens: {total_tokens:,} ({total_input:,} in / {total_output:,} out)")
        print(f"  Estimated cost: ${approx_cost:.2f}")
    else:
        print(f"  Tokens so far: {total_tokens:,} (~${approx_cost:.2f})")


def _handle_utility_command() -> bool:
    """Handle utility subcommands (apply, status). Returns True if handled."""
    if len(sys.argv) <= 1:
        return False

    cmd = sys.argv[1]
    if cmd == "apply":
        from darwinian_evolver.apply import main as apply_main

        apply_main(sys.argv[2:])
        return True
    elif cmd == "status":
        from darwinian_evolver.status import main as status_main

        status_main(sys.argv[2:])
        return True
    return False


if __name__ == "__main__":
    if _handle_utility_command():
        sys.exit(0)

    args = parse_args()

    # Select the specified problem
    if args.problem in AVAILABLE_PROBLEMS:
        problem = AVAILABLE_PROBLEMS[args.problem]()
    elif args.problem in CONFIGURABLE_PROBLEMS:
        if args.problem == "repo_task":
            if not args.repo_root:
                print("Error: --repo_root is required for repo_task problem")
                sys.exit(1)
            if not args.task:
                print("Error: --task is required for repo_task problem")
                sys.exit(1)
            if not args.test_command:
                print("Error: --test_command is required for repo_task problem")
                sys.exit(1)
            problem = CONFIGURABLE_PROBLEMS[args.problem](
                repo_root=args.repo_root,
                task=args.task,
                test_command=args.test_command,
                setup_command=args.setup_command,
                files=args.files,
                agent_model=args.agent_model,
                agent_max_turns=args.agent_max_turns,
                test_timeout=args.test_timeout,
            )
        elif args.problem == "spec_task":
            if not args.repo_root:
                print("Error: --repo_root is required for spec_task problem")
                sys.exit(1)
            spec = args.spec
            if not spec and args.spec_file:
                spec = args.spec_file.read_text()
            if not spec:
                print("Error: --spec or --spec_file is required for spec_task problem")
                sys.exit(1)
            problem = CONFIGURABLE_PROBLEMS[args.problem](
                repo_root=args.repo_root,
                spec=spec,
                task=args.task or "",
                validation_command=args.validation_command,
                files=args.files,
                agent_model=args.agent_model,
                judge_model=args.judge_model,
                agent_max_turns=args.agent_max_turns,
            )
        else:
            print(f"Error: No configuration handler for problem '{args.problem}'")
            sys.exit(1)
    else:
        print(f"Error: Unknown problem '{args.problem}'")
        sys.exit(1)

    if args.batch_size > 1 and not any(mutator.supports_batch_mutation for mutator in problem.mutators):
        print(
            "Warning: Batch size is set to greater than 1, but no mutators of this problem support batch mutation. Batch size will have no effect."
        )

    # Validate fixed_children_per_generation if provided
    fixed_children_per_generation = args.fixed_children_per_generation
    if fixed_children_per_generation is not None:
        if any(x <= 0 for x in fixed_children_per_generation):
            print("Error: All values in --fixed_children_per_generation must be positive")
            sys.exit(1)

    hyperparameter_config = build_hyperparameter_config_from_args(args)

    # Setup output directories
    snapshot_dir = None
    json_log_file = None
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_log_file = args.output_dir / "results.jsonl"
        snapshot_dir = args.output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        problem.evaluator.set_output_dir(str(args.output_dir))
    elif args.s3_dir:
        # Create temp file for results.jsonl when uploading to S3 but no local output_dir
        json_log_file = Path(tempfile.mktemp(suffix=".jsonl"))

    if args.s3_dir:
        problem.evaluator.set_s3_dir(str(args.s3_dir))

    evolve_loop = EvolveProblemLoop(
        problem,
        learning_log_view_type=parse_learning_log_view_type(hyperparameter_config.learning_log_view_type),
        num_parents_per_iteration=hyperparameter_config.num_parents_per_iteration,
        mutator_concurrency=args.mutator_concurrency,
        evaluator_concurrency=args.evaluator_concurrency,
        snapshot_to_resume_from=args.resume_from_snapshot.read_bytes() if args.resume_from_snapshot else None,
        fixed_midpoint_score=hyperparameter_config.fixed_midpoint_score,
        midpoint_score_percentile=hyperparameter_config.midpoint_score_percentile,
        sharpness=hyperparameter_config.sharpness,
        novelty_weight=hyperparameter_config.novelty_weight,
        batch_size=hyperparameter_config.batch_size,
        should_verify_mutations=hyperparameter_config.verify_mutations,
        fixed_children_per_generation=fixed_children_per_generation,
        use_process_pool_executors=args.use_process_pool,
    )

    if args.resume_from_snapshot:
        print(f"Resuming from snapshot: {args.resume_from_snapshot}")
    else:
        print("Evaluating initial organism...")

    early_stopped = False
    for snapshot in evolve_loop.run(num_iterations=args.num_iterations):
        print_snapshot_summary(snapshot)

        # Print cost tracking info
        _print_cost_summary(problem)

        if snapshot_dir:
            snapshot_file = snapshot_dir / f"iteration_{snapshot.iteration}.pkl"
            if snapshot_file.exists() and not args.overwrite_snapshots:
                print(f"Snapshot {snapshot_file} already exists. Use --overwrite_snapshots to overwrite.")
                sys.exit(1)

            with snapshot_file.open("wb") as file:
                file.write(snapshot.snapshot)

        if args.s3_dir:
            upload_bytes_to_s3(
                snapshot.snapshot,
                str(args.s3_dir),
                f"snapshots/iteration_{snapshot.iteration}.pkl",
            )

        if json_log_file:
            with json_log_file.open("a") as file:
                log_dict = {
                    "iteration": snapshot.iteration,
                    "population": snapshot.population_json_log,
                    "verification_failures": snapshot.population_json_log.get("organisms_failed_verification", []),
                }
                file.write(json.dumps(log_dict) + "\n")

        # Early stopping check
        if args.early_stop_score is not None:
            best_score = snapshot.best_organism_result[1].score
            if best_score >= args.early_stop_score:
                print(f"\n  Early stopping: best score {best_score:.4f} >= threshold {args.early_stop_score}")
                early_stopped = True
                break

    # Final cost summary
    _print_cost_summary(problem, final=True)

    if early_stopped and args.output_dir:
        print(f"\n  Results saved to: {args.output_dir}")
        print(f"  Apply best changes: python -m darwinian_evolver apply --output_dir {args.output_dir}")

    if args.s3_dir:
        upload_file_to_s3_fixed_path(json_log_file, str(args.s3_dir), "results.jsonl")
