"""Factory function to create a spec_task Problem from CLI arguments."""

from __future__ import annotations

from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.repo_task import auto_detect_files
from darwinian_evolver.problems.spec_task import SpecTaskEvaluationResult
from darwinian_evolver.problems.spec_task import SpecTaskEvaluator
from darwinian_evolver.problems.spec_task import SpecTaskFailureCase
from darwinian_evolver.problems.spec_task import SpecTaskOrganism
from darwinian_evolver.problems.spec_task_agent import AgenticSpecMutator


def make_spec_task_problem(
    repo_root: str,
    spec: str,
    task: str = "",
    validation_command: str | None = None,
    files: list[str] | None = None,
    agent_model: str = "claude-sonnet-4-20250514",
    judge_model: str = "claude-sonnet-4-20250514",
    agent_max_turns: int = 25,
) -> Problem[SpecTaskOrganism, SpecTaskEvaluationResult, SpecTaskFailureCase]:
    """Create a spec_task Problem configured from CLI arguments."""
    if files:
        files_to_capture = files
    else:
        effective_task = f"{task}\n\nSpec: {spec}" if task else spec
        print("Auto-detecting relevant files for spec...")
        files_to_capture = auto_detect_files(repo_root, effective_task)
        print(f"Detected {len(files_to_capture)} files: {', '.join(files_to_capture[:10])}")

    initial_organism = SpecTaskOrganism.make_initial_organism_from_repo(
        repo_root=repo_root,
        files_to_capture=files_to_capture,
        task_description=task,
        spec=spec,
    )

    evaluator = SpecTaskEvaluator(
        judge_model=judge_model,
        validation_command=validation_command,
    )

    mutator = AgenticSpecMutator(
        model=agent_model,
        max_turns=agent_max_turns,
    )

    return Problem(
        initial_organism=initial_organism,
        evaluator=evaluator,
        mutators=[mutator],
    )
