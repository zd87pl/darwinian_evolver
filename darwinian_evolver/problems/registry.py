from typing import Callable

from darwinian_evolver.problem import Problem
from darwinian_evolver.problems.circle_packing import make_circle_packing_problem
from darwinian_evolver.problems.multiplication_verifier import make_multiplication_verifier_problem
from darwinian_evolver.problems.parrot import make_parrot_problem

AVAILABLE_PROBLEMS: dict[str, Callable[[], Problem]] = {
    "parrot": make_parrot_problem,
    "circle_packing": make_circle_packing_problem,
    "multiplication_verifier": make_multiplication_verifier_problem,
}

# Problems that require extra CLI args and are constructed via make_*_problem(**kwargs)
CONFIGURABLE_PROBLEMS: dict[str, Callable[..., Problem]] = {}


def register_repo_task() -> None:
    """Register the repo_task problem (lazy import to avoid import cost when not used)."""
    from darwinian_evolver.problems.repo_task_factory import make_repo_task_problem

    CONFIGURABLE_PROBLEMS["repo_task"] = make_repo_task_problem
