"""Factory function to create a repo_task Problem from CLI arguments."""

from __future__ import annotations

from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.repo_task import RepoTaskEvaluationResult
from darwinian_evolver.problems.repo_task import RepoTaskEvaluator
from darwinian_evolver.problems.repo_task import RepoTaskFailureCase
from darwinian_evolver.problems.repo_task import RepoTaskOrganism
from darwinian_evolver.problems.repo_task import auto_detect_files
from darwinian_evolver.problems.repo_task_agent import AgenticRepoMutator


def make_repo_task_problem(
    repo_root: str,
    task: str,
    test_command: str,
    setup_command: str | None = None,
    files: list[str] | None = None,
    agent_model: str = "claude-sonnet-4-20250514",
    agent_max_turns: int = 25,
    test_timeout: int = 300,
) -> Problem[RepoTaskOrganism, RepoTaskEvaluationResult, RepoTaskFailureCase]:
    """Create a repo_task Problem configured from CLI arguments."""
    # Determine which files to track
    if files:
        files_to_capture = files
    else:
        print(f"Auto-detecting relevant files for task: {task[:80]}...")
        files_to_capture = auto_detect_files(repo_root, task)
        print(f"Detected {len(files_to_capture)} files: {', '.join(files_to_capture[:10])}")
        if len(files_to_capture) > 10:
            print(f"  ... and {len(files_to_capture) - 10} more")

    initial_organism = RepoTaskOrganism.make_initial_organism_from_repo(
        repo_root=repo_root,
        files_to_capture=files_to_capture,
        task_description=task,
    )

    evaluator = RepoTaskEvaluator(
        test_command=test_command,
        setup_command=setup_command,
        timeout=test_timeout,
    )

    mutator = AgenticRepoMutator(
        model=agent_model,
        max_turns=agent_max_turns,
    )

    return Problem(
        initial_organism=initial_organism,
        evaluator=evaluator,
        mutators=[mutator],
    )
