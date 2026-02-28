"""
Problem type for evolving code in a git repository to satisfy a task (typically fixing failing tests).

The evaluator runs a test command and parses the output to identify failures.
The agentic mutator (in repo_task_agent.py) uses multi-turn tool-use to explore and modify the code.
"""

from __future__ import annotations

import os
import re
import subprocess

from pydantic import computed_field

from darwinian_evolver.git_based_problem import GitBasedOrganism
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator


class RepoTaskOrganism(GitBasedOrganism):
    task_description: str = ""

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        props: dict[str, str | float] = {}
        if self.from_change_summary:
            props["change_summary"] = self.from_change_summary[:200]
        props["num_files"] = len(self.file_contents)
        return props


class RepoTaskFailureCase(EvaluationFailureCase):
    test_name: str
    error_output: str = ""

    @computed_field
    @property
    def data_point_id(self) -> str:
        return self.test_name


class RepoTaskEvaluationResult(EvaluationResult):
    raw_output: str = ""
    num_passed: int = 0
    num_failed: int = 0
    num_errors: int = 0
    num_total: int = 0

    def format_observed_outcome(self, parent_result: EvaluationResult | None, ndigits: int = 2) -> str:
        outcome = f"{self.num_passed}/{self.num_total} tests passed (score: {round(self.score, ndigits)})."
        if parent_result is not None and isinstance(parent_result, RepoTaskEvaluationResult):
            parent_passed = parent_result.num_passed
            delta = self.num_passed - parent_passed
            if delta > 0:
                outcome += f" +{delta} tests fixed vs parent."
            elif delta < 0:
                outcome += f" {delta} tests regressed vs parent."
            else:
                outcome += " No change from parent."
        return outcome

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        return {
            "passed": self.num_passed,
            "failed": self.num_failed,
            "errors": self.num_errors,
            "total": self.num_total,
        }


class RepoTaskEvaluator(Evaluator[RepoTaskOrganism, RepoTaskEvaluationResult, RepoTaskFailureCase]):
    def __init__(
        self,
        test_command: str,
        setup_command: str | None = None,
        timeout: int = 300,
    ) -> None:
        self._test_command = test_command
        self._setup_command = setup_command
        self._timeout = timeout

    def evaluate(self, organism: RepoTaskOrganism) -> RepoTaskEvaluationResult:
        with organism.build_repo() as temp_dir:
            if self._setup_command:
                try:
                    subprocess.run(
                        self._setup_command,
                        shell=True,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    return RepoTaskEvaluationResult(
                        score=0.0,
                        trainable_failure_cases=[
                            RepoTaskFailureCase(
                                test_name="__setup__",
                                error_output="Setup command timed out after 120 seconds",
                                failure_type="setup_error",
                            )
                        ],
                        raw_output="Setup timed out",
                        is_viable=False,
                    )

            try:
                result = subprocess.run(
                    self._test_command,
                    shell=True,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired:
                return RepoTaskEvaluationResult(
                    score=0.0,
                    trainable_failure_cases=[
                        RepoTaskFailureCase(
                            test_name="__timeout__",
                            error_output=f"Test command timed out after {self._timeout} seconds",
                            failure_type="timeout",
                        )
                    ],
                    raw_output="Test command timed out",
                    is_viable=False,
                )

            combined_output = result.stdout + "\n" + result.stderr
            failures = self._parse_test_output(combined_output, result.returncode)

            # Try to extract test counts from output
            num_passed, num_failed, num_errors, num_total = self._extract_test_counts(combined_output, failures)

            if num_total > 0:
                score = num_passed / num_total
            elif result.returncode == 0:
                score = 1.0
                num_total = 1
                num_passed = 1
            else:
                score = 0.0
                num_total = max(1, len(failures))

            is_viable = score > 0.0 or result.returncode == 0

            return RepoTaskEvaluationResult(
                score=score,
                trainable_failure_cases=failures,
                raw_output=combined_output[-5000:],  # Keep last 5000 chars
                num_passed=num_passed,
                num_failed=num_failed,
                num_errors=num_errors,
                num_total=num_total,
                is_viable=is_viable,
            )

    def verify_mutation(self, organism: RepoTaskOrganism) -> bool:
        """Quick check: does at least one previously-failing test now pass?"""
        if not organism.from_failure_cases:
            return True

        with organism.build_repo() as temp_dir:
            if self._setup_command:
                try:
                    subprocess.run(
                        self._setup_command,
                        shell=True,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    return False

            try:
                result = subprocess.run(
                    self._test_command,
                    shell=True,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired:
                return False

            combined_output = result.stdout + "\n" + result.stderr
            current_failures = self._parse_test_output(combined_output, result.returncode)
            current_failure_names = {f.test_name for f in current_failures}

            # Check if at least one previously-failing test is no longer failing
            for prev_failure in organism.from_failure_cases:
                if isinstance(prev_failure, RepoTaskFailureCase):
                    if prev_failure.test_name not in current_failure_names:
                        return True

            return False

    def _parse_test_output(self, output: str, returncode: int) -> list[RepoTaskFailureCase]:
        """Parse test output to extract individual failure cases."""
        failures: list[RepoTaskFailureCase] = []

        # Try pytest format first
        pytest_failures = self._parse_pytest_output(output)
        if pytest_failures:
            return pytest_failures

        # Try unittest format
        unittest_failures = self._parse_unittest_output(output)
        if unittest_failures:
            return unittest_failures

        # Generic fallback: if returncode != 0, create a single failure
        if returncode != 0:
            # Try to extract meaningful error lines
            error_lines = []
            for line in output.split("\n"):
                line_stripped = line.strip()
                if any(keyword in line_stripped.upper() for keyword in ["ERROR", "FAIL", "ASSERT", "TRACEBACK"]):
                    error_lines.append(line_stripped)

            failures.append(
                RepoTaskFailureCase(
                    test_name="__generic_failure__",
                    error_output="\n".join(error_lines[-20:]) if error_lines else output[-2000:],
                    failure_type="generic_failure",
                )
            )

        return failures

    def _parse_pytest_output(self, output: str) -> list[RepoTaskFailureCase]:
        """Parse pytest --tb=short style output."""
        failures: list[RepoTaskFailureCase] = []

        # Match pytest FAILED lines: "FAILED path/to/test.py::TestClass::test_method"
        failed_pattern = re.compile(r"FAILED\s+(\S+)")
        for match in failed_pattern.finditer(output):
            test_name = match.group(1)
            # Extract traceback for this test if available
            error_output = self._extract_pytest_traceback(output, test_name)
            failures.append(
                RepoTaskFailureCase(
                    test_name=test_name,
                    error_output=error_output[:2000],
                    failure_type="test_failure",
                )
            )

        # Match pytest ERROR lines: "ERROR path/to/test.py::test_method"
        error_pattern = re.compile(r"ERROR\s+(\S+)")
        for match in error_pattern.finditer(output):
            test_name = match.group(1)
            failures.append(
                RepoTaskFailureCase(
                    test_name=test_name,
                    error_output=self._extract_pytest_traceback(output, test_name)[:2000],
                    failure_type="error",
                )
            )

        return failures

    def _extract_pytest_traceback(self, output: str, test_name: str) -> str:
        """Extract the traceback section for a specific test from pytest output."""
        # Look for the test's section in the output
        # pytest sections start with "_____ test_name _____" or "FAILED test_name"
        test_base = test_name.split("::")[-1] if "::" in test_name else test_name
        lines = output.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if test_base in line and ("_____" in line or "FAILED" in line):
                in_section = True
                section_lines = [line]
                continue
            if in_section:
                if line.startswith("_____") or line.startswith("====="):
                    break
                section_lines.append(line)

        return "\n".join(section_lines) if section_lines else ""

    def _parse_unittest_output(self, output: str) -> list[RepoTaskFailureCase]:
        """Parse unittest-style output."""
        failures: list[RepoTaskFailureCase] = []

        # Match "FAIL: test_name (module.TestClass)" or "ERROR: test_name (module.TestClass)"
        pattern = re.compile(r"(FAIL|ERROR):\s+(\S+)\s+\(([^)]+)\)")
        for match in pattern.finditer(output):
            failure_type = "test_failure" if match.group(1) == "FAIL" else "error"
            test_method = match.group(2)
            test_class = match.group(3)
            test_name = f"{test_class}.{test_method}"
            failures.append(
                RepoTaskFailureCase(
                    test_name=test_name,
                    error_output="",
                    failure_type=failure_type,
                )
            )

        return failures

    def _extract_test_counts(self, output: str, failures: list[RepoTaskFailureCase]) -> tuple[int, int, int, int]:
        """Extract test counts from test output."""
        num_failed = 0
        num_errors = 0
        num_passed = 0
        num_total = 0

        # Try pytest summary line: "X passed, Y failed, Z errors"
        pytest_summary = re.search(r"=+\s*(.*?)\s*=+\s*$", output, re.MULTILINE)
        if pytest_summary:
            summary_text = pytest_summary.group(1)
            passed_match = re.search(r"(\d+)\s+passed", summary_text)
            failed_match = re.search(r"(\d+)\s+failed", summary_text)
            error_match = re.search(r"(\d+)\s+error", summary_text)

            if passed_match:
                num_passed = int(passed_match.group(1))
            if failed_match:
                num_failed = int(failed_match.group(1))
            if error_match:
                num_errors = int(error_match.group(1))

            num_total = num_passed + num_failed + num_errors
            if num_total > 0:
                return num_passed, num_failed, num_errors, num_total

        # Try unittest summary: "Ran X tests"
        unittest_summary = re.search(r"Ran\s+(\d+)\s+test", output)
        if unittest_summary:
            num_total = int(unittest_summary.group(1))
            num_failed = len([f for f in failures if f.failure_type == "test_failure"])
            num_errors = len([f for f in failures if f.failure_type == "error"])
            num_passed = num_total - num_failed - num_errors
            return max(0, num_passed), num_failed, num_errors, num_total

        # Fallback: count from parsed failures
        num_failed = len(failures)
        return num_passed, num_failed, num_errors, num_total


def auto_detect_files(
    repo_root: str,
    task_description: str,
    git_hash: str | None = None,
) -> list[str]:
    """
    Use an LLM to identify which files in the repo are relevant to the task.

    Falls back to capturing all non-binary files under a size threshold if the LLM call fails.
    """
    import contextlib

    from anthropic import Anthropic

    with contextlib.chdir(repo_root):
        if git_hash is None:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

        file_list = subprocess.check_output(["git", "ls-files"]).decode("utf-8").strip()

    prompt = f"""You are analyzing a git repository to determine which files are relevant to a coding task.

Repository files:
{file_list}

Task:
{task_description}

Return ONLY a JSON array of file paths that are most likely to need modification or are important context for understanding the task. Include:
- Files that likely need to be modified to complete the task
- Test files related to the task
- Key imports/dependencies of the files that need modification

Keep the list focused — typically 3-15 files. Do not include configuration files, lock files, or unrelated modules unless specifically relevant.

Return ONLY the JSON array, no other text. Example: ["src/auth.py", "tests/test_auth.py", "src/models/user.py"]"""

    try:
        client = Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        import json

        text = response.content[0].text.strip()
        # Extract JSON array from response
        start = text.index("[")
        end = text.rindex("]") + 1
        files = json.loads(text[start:end])

        # Validate that files exist
        all_files = set(file_list.split("\n"))
        valid_files = [f for f in files if f in all_files]
        if valid_files:
            return valid_files
    except Exception:
        pass

    # Fallback: capture all Python files and common source files
    return _fallback_file_detection(file_list)


def _fallback_file_detection(file_list: str) -> list[str]:
    """Fallback file detection: pick source files under a size limit."""
    source_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".cpp", ".h"}
    files = []
    for f in file_list.split("\n"):
        f = f.strip()
        if not f:
            continue
        _, ext = os.path.splitext(f)
        if ext in source_extensions:
            files.append(f)
    return files[:50]  # Cap at 50 files
