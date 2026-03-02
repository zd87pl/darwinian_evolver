"""
Spec-driven problem type for greenfield code generation.

Instead of test commands, the evaluator uses an LLM-as-judge to score how well
the generated code meets a natural language specification. This enables evolving
code for tasks where no test suite exists yet.
"""

from __future__ import annotations

import subprocess

from anthropic import Anthropic
from pydantic import computed_field

from darwinian_evolver.git_based_problem import GitBasedOrganism
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator


class SpecTaskOrganism(GitBasedOrganism):
    task_description: str = ""
    spec: str = ""

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        props: dict[str, str | float] = {}
        if self.from_change_summary:
            props["change_summary"] = self.from_change_summary[:200]
        props["num_files"] = len(self.file_contents)
        return props


class SpecTaskFailureCase(EvaluationFailureCase):
    criterion: str
    feedback: str = ""

    def __init__(self, **data):
        if "data_point_id" not in data:
            data["data_point_id"] = data.get("criterion", "")[:80]
        super().__init__(**data)


class SpecTaskEvaluationResult(EvaluationResult):
    judge_reasoning: str = ""
    criteria_scores: dict[str, float] = {}

    def format_observed_outcome(self, parent_result: EvaluationResult | None, ndigits: int = 2) -> str:
        outcome = f"Score: {round(self.score, ndigits)}/1.0."
        if self.criteria_scores:
            details = ", ".join(f"{k}: {round(v, 1)}/10" for k, v in self.criteria_scores.items())
            outcome += f" Criteria: {details}."
        if parent_result is not None:
            delta = round(self.score - parent_result.score, ndigits)
            if delta > 0:
                outcome += f" +{delta} improvement."
            elif delta < 0:
                outcome += f" {delta} regression."
        return outcome

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        props: dict[str, str | float] = {"score": self.score}
        props.update({k: v for k, v in self.criteria_scores.items()})
        return props


JUDGE_SYSTEM_PROMPT = """You are a code review judge. You evaluate code against a specification and provide structured feedback.

You must respond with EXACTLY this JSON format (no other text):
{
    "overall_score": <float 0.0-1.0>,
    "criteria": {
        "<criterion_name>": {
            "score": <int 0-10>,
            "feedback": "<specific feedback>"
        }
    },
    "reasoning": "<1-3 sentence overall assessment>"
}

Criteria to evaluate:
- correctness: Does the code correctly implement what the spec asks for?
- completeness: Are all parts of the spec addressed?
- code_quality: Is the code clean, readable, and well-structured?
- edge_cases: Does it handle edge cases and errors appropriately?
"""


class SpecTaskEvaluator(Evaluator[SpecTaskOrganism, SpecTaskEvaluationResult, SpecTaskFailureCase]):
    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-20250514",
        validation_command: str | None = None,
        validation_timeout: int = 60,
    ) -> None:
        self._judge_model = judge_model
        self._validation_command = validation_command
        self._validation_timeout = validation_timeout
        self._client = Anthropic()

    def evaluate(self, organism: SpecTaskOrganism) -> SpecTaskEvaluationResult:
        # Optional: run a validation command first (e.g., syntax check, type check)
        if self._validation_command:
            validation_result = self._run_validation(organism)
            if validation_result is not None:
                return validation_result

        # Build the code snapshot for the judge
        code_snapshot = self._build_code_snapshot(organism)

        prompt = f"""Evaluate the following code against this specification:

## Specification
{organism.spec}

## Task Description
{organism.task_description}

## Code
{code_snapshot}

Evaluate how well this code meets the specification."""

        try:
            response = self._client.messages.create(
                model=self._judge_model,
                max_tokens=2048,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_judge_response(response.content[0].text)
        except Exception as e:
            return SpecTaskEvaluationResult(
                score=0.0,
                trainable_failure_cases=[
                    SpecTaskFailureCase(
                        criterion="judge_error",
                        feedback=f"Judge evaluation failed: {e}",
                        failure_type="judge_error",
                    )
                ],
                judge_reasoning=f"Judge error: {e}",
                is_viable=False,
            )

    def _run_validation(self, organism: SpecTaskOrganism) -> SpecTaskEvaluationResult | None:
        """Run a validation command. Returns an error result if validation fails, None if it passes."""
        with organism.build_repo() as temp_dir:
            try:
                result = subprocess.run(
                    self._validation_command,
                    shell=True,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._validation_timeout,
                )
                if result.returncode != 0:
                    error_output = (result.stdout + "\n" + result.stderr)[-2000:]
                    return SpecTaskEvaluationResult(
                        score=0.0,
                        trainable_failure_cases=[
                            SpecTaskFailureCase(
                                criterion="validation",
                                feedback=f"Validation command failed:\n{error_output}",
                                failure_type="validation_error",
                            )
                        ],
                        judge_reasoning=f"Validation failed: {error_output[:500]}",
                        is_viable=False,
                    )
            except subprocess.TimeoutExpired:
                return SpecTaskEvaluationResult(
                    score=0.0,
                    trainable_failure_cases=[
                        SpecTaskFailureCase(
                            criterion="validation",
                            feedback="Validation command timed out",
                            failure_type="timeout",
                        )
                    ],
                    judge_reasoning="Validation timed out",
                    is_viable=False,
                )
        return None

    def _build_code_snapshot(self, organism: SpecTaskOrganism) -> str:
        parts = []
        for path in sorted(organism.file_contents.keys()):
            content = organism.file_contents[path]
            lang = path.rsplit(".", 1)[-1] if "." in path else ""
            parts.append(f"### {path}\n```{lang}\n{content}\n```")
        return "\n\n".join(parts)

    def _parse_judge_response(self, response_text: str) -> SpecTaskEvaluationResult:
        import json

        try:
            # Extract JSON from response
            text = response_text.strip()
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            return SpecTaskEvaluationResult(
                score=0.3,
                trainable_failure_cases=[
                    SpecTaskFailureCase(
                        criterion="judge_parse_error",
                        feedback=f"Could not parse judge response: {response_text[:500]}",
                        failure_type="judge_error",
                    )
                ],
                judge_reasoning="Judge response was not valid JSON",
                is_viable=True,
            )

        overall_score = float(data.get("overall_score", 0.0))
        overall_score = max(0.0, min(1.0, overall_score))
        reasoning = data.get("reasoning", "")
        criteria = data.get("criteria", {})

        criteria_scores = {}
        failure_cases = []
        for criterion_name, criterion_data in criteria.items():
            score = criterion_data.get("score", 0)
            feedback = criterion_data.get("feedback", "")
            criteria_scores[criterion_name] = float(score)
            # Anything below 7/10 is a failure case the mutator can work on
            if score < 7:
                failure_cases.append(
                    SpecTaskFailureCase(
                        criterion=criterion_name,
                        feedback=feedback,
                        failure_type=criterion_name,
                    )
                )

        return SpecTaskEvaluationResult(
            score=overall_score,
            trainable_failure_cases=failure_cases,
            judge_reasoning=reasoning,
            criteria_scores=criteria_scores,
            is_viable=overall_score > 0.0,
        )
