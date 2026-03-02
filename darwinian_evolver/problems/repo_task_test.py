"""
End-to-end integration tests for the repo_task problem type.

Tests the evaluator, test output parsing, and the full mutation pipeline
against a small synthetic git repository with deliberately failing tests.
"""

import os
import subprocess
import sys
import tempfile

import pytest

from darwinian_evolver.problems.repo_task import RepoTaskEvaluationResult
from darwinian_evolver.problems.repo_task import RepoTaskEvaluator
from darwinian_evolver.problems.repo_task import RepoTaskFailureCase
from darwinian_evolver.problems.repo_task import RepoTaskOrganism

# Build a pytest command that uses the same Python interpreter running these tests
_PYTEST_CMD = f"{sys.executable} -m pytest tests/ --tb=short"


def _create_test_repo(temp_dir: str) -> str:
    """Create a small git repo with a source file and test file, where one test fails."""
    repo_path = os.path.join(temp_dir, "test_repo")
    os.makedirs(repo_path)

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    # Disable GPG signing for test commits
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create source file with a bug
    os.makedirs(os.path.join(repo_path, "src"), exist_ok=True)
    with open(os.path.join(repo_path, "src", "math_utils.py"), "w") as f:
        f.write(
            "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a + b  # BUG: should be a * b\n"
        )

    # Create test file
    os.makedirs(os.path.join(repo_path, "tests"), exist_ok=True)
    with open(os.path.join(repo_path, "tests", "test_math.py"), "w") as f:
        f.write(
            "import sys\nimport os\n"
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n"
            "from src.math_utils import add, multiply\n\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n\n\n"
            "def test_multiply():\n    assert multiply(3, 4) == 12\n\n\n"
            "def test_multiply_zero():\n    assert multiply(5, 0) == 0\n"
        )

    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


# --- Test output parsing ---


class TestRepoTaskEvaluator:
    def test_parse_pytest_output_failures(self):
        evaluator = RepoTaskEvaluator(test_command="pytest")
        output = """
============================= test session starts ==============================
collected 3 items

tests/test_math.py .FF                                                   [100%]

=================================== FAILURES ===================================
_____________________________ test_multiply ______________________________

    def test_multiply():
>       assert multiply(3, 4) == 12
E       assert 7 == 12

tests/test_math.py:10: AssertionError
_____________________________ test_multiply_zero _____________________________

    def test_multiply_zero():
>       assert multiply(5, 0) == 0
E       assert 5 == 0

tests/test_math.py:14: AssertionError
=========================== short test summary info ============================
FAILED tests/test_math.py::test_multiply
FAILED tests/test_math.py::test_multiply_zero
========================= 1 passed, 2 failed in 0.02s =========================
"""
        failures = evaluator._parse_test_output(output, returncode=1)
        assert len(failures) == 2
        assert failures[0].test_name == "tests/test_math.py::test_multiply"
        assert failures[1].test_name == "tests/test_math.py::test_multiply_zero"

    def test_parse_pytest_output_all_pass(self):
        evaluator = RepoTaskEvaluator(test_command="pytest")
        output = """
============================= test session starts ==============================
collected 3 items

tests/test_math.py ...                                                   [100%]

============================== 3 passed in 0.01s ===============================
"""
        failures = evaluator._parse_test_output(output, returncode=0)
        assert len(failures) == 0

    def test_extract_test_counts_pytest(self):
        evaluator = RepoTaskEvaluator(test_command="pytest")
        output = "========================= 1 passed, 2 failed in 0.02s ========================="
        num_passed, num_failed, num_errors, num_total = evaluator._extract_test_counts(output, [])
        assert num_passed == 1
        assert num_failed == 2
        assert num_total == 3

    def test_extract_test_counts_unittest(self):
        evaluator = RepoTaskEvaluator(test_command="python -m unittest")
        output = """
FAIL: test_multiply (tests.test_math.TestMath)
----------------------------------------------------------------------
Ran 3 tests in 0.001s

FAILED (failures=2)
"""
        failures = [
            RepoTaskFailureCase(test_name="t1", failure_type="test_failure"),
            RepoTaskFailureCase(test_name="t2", failure_type="test_failure"),
        ]
        num_passed, num_failed, num_errors, num_total = evaluator._extract_test_counts(output, failures)
        assert num_total == 3
        assert num_failed == 2
        assert num_passed == 1

    def test_generic_failure_on_nonzero_exit(self):
        evaluator = RepoTaskEvaluator(test_command="pytest")
        output = "ImportError: No module named 'foo'\nTraceback (most recent call last):\n  File ...\n"
        failures = evaluator._parse_test_output(output, returncode=1)
        assert len(failures) == 1
        assert failures[0].test_name == "__generic_failure__"


# --- Evaluation result formatting ---


class TestRepoTaskEvaluationResult:
    def test_format_observed_outcome_improvement(self):
        parent = RepoTaskEvaluationResult(
            score=0.33,
            trainable_failure_cases=[],
            num_passed=1,
            num_failed=2,
            num_total=3,
        )
        child = RepoTaskEvaluationResult(
            score=1.0,
            trainable_failure_cases=[],
            num_passed=3,
            num_failed=0,
            num_total=3,
        )
        outcome = child.format_observed_outcome(parent)
        assert "+2 tests fixed" in outcome

    def test_format_observed_outcome_regression(self):
        parent = RepoTaskEvaluationResult(
            score=1.0,
            trainable_failure_cases=[],
            num_passed=3,
            num_failed=0,
            num_total=3,
        )
        child = RepoTaskEvaluationResult(
            score=0.33,
            trainable_failure_cases=[],
            num_passed=1,
            num_failed=2,
            num_total=3,
        )
        outcome = child.format_observed_outcome(parent)
        assert "-2 tests regressed" in outcome


# --- Organism + evaluator integration ---


class TestRepoTaskIntegration:
    @pytest.fixture
    def test_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = _create_test_repo(temp_dir)
            yield repo_path

    def test_create_organism_from_repo(self, test_repo):
        organism = RepoTaskOrganism.make_initial_organism_from_repo(
            repo_root=test_repo,
            files_to_capture=["src/math_utils.py", "tests/test_math.py"],
            task_description="Fix the multiply function",
        )
        assert "src/math_utils.py" in organism.file_contents
        assert "tests/test_math.py" in organism.file_contents
        assert "def add" in organism.file_contents["src/math_utils.py"]
        assert organism.task_description == "Fix the multiply function"

    def test_evaluate_organism_with_failures(self, test_repo):
        organism = RepoTaskOrganism.make_initial_organism_from_repo(
            repo_root=test_repo,
            files_to_capture=["src/math_utils.py", "tests/test_math.py"],
            task_description="Fix the multiply function",
        )
        evaluator = RepoTaskEvaluator(test_command=_PYTEST_CMD)
        result = evaluator.evaluate(organism)

        assert isinstance(result, RepoTaskEvaluationResult)
        assert result.score < 1.0  # Not all tests pass
        assert result.num_failed > 0
        assert len(result.trainable_failure_cases) > 0

    def test_evaluate_fixed_organism(self, test_repo):
        organism = RepoTaskOrganism.make_initial_organism_from_repo(
            repo_root=test_repo,
            files_to_capture=["src/math_utils.py", "tests/test_math.py"],
            task_description="Fix the multiply function",
        )
        # Manually fix the bug
        fixed_contents = dict(organism.file_contents)
        fixed_contents["src/math_utils.py"] = fixed_contents["src/math_utils.py"].replace(
            "return a + b  # BUG: should be a * b",
            "return a * b",
        )
        fixed_organism = RepoTaskOrganism(
            repo_root=organism.repo_root,
            git_hash=organism.git_hash,
            file_contents=fixed_contents,
            task_description=organism.task_description,
        )

        evaluator = RepoTaskEvaluator(test_command=_PYTEST_CMD)
        result = evaluator.evaluate(fixed_organism)

        assert result.score == 1.0
        assert result.num_failed == 0
        assert len(result.trainable_failure_cases) == 0

    def test_verify_mutation_success(self, test_repo):
        organism = RepoTaskOrganism.make_initial_organism_from_repo(
            repo_root=test_repo,
            files_to_capture=["src/math_utils.py", "tests/test_math.py"],
            task_description="Fix the multiply function",
        )
        evaluator = RepoTaskEvaluator(test_command=_PYTEST_CMD)
        initial_result = evaluator.evaluate(organism)

        # Create a fixed organism with from_failure_cases set
        fixed_contents = dict(organism.file_contents)
        fixed_contents["src/math_utils.py"] = fixed_contents["src/math_utils.py"].replace(
            "return a + b  # BUG: should be a * b",
            "return a * b",
        )
        fixed_organism = RepoTaskOrganism(
            repo_root=organism.repo_root,
            git_hash=organism.git_hash,
            file_contents=fixed_contents,
            task_description=organism.task_description,
            from_failure_cases=initial_result.trainable_failure_cases,
        )

        assert evaluator.verify_mutation(fixed_organism) is True

    def test_build_repo_supports_new_files(self, test_repo):
        organism = RepoTaskOrganism.make_initial_organism_from_repo(
            repo_root=test_repo,
            files_to_capture=["src/math_utils.py"],
            task_description="test",
        )
        # Add a new file to the organism
        new_contents = dict(organism.file_contents)
        new_contents["src/new_module.py"] = "def hello():\n    return 'hello'\n"
        new_organism = RepoTaskOrganism(
            repo_root=organism.repo_root,
            git_hash=organism.git_hash,
            file_contents=new_contents,
            task_description="test",
        )

        with new_organism.build_repo() as temp_dir:
            assert os.path.exists(os.path.join(temp_dir, "src", "new_module.py"))
            with open(os.path.join(temp_dir, "src", "new_module.py")) as f:
                assert f.read() == "def hello():\n    return 'hello'\n"


# --- Agent tool tests (unit-level, no API calls) ---


class TestAgentTools:
    @pytest.fixture
    def agent(self):
        from darwinian_evolver.problems.repo_task_agent import AgenticRepoMutator

        return AgenticRepoMutator()

    @pytest.fixture
    def temp_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "src"))
            with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
                f.write("def greet(name):\n    return f'Hello, {name}!'\n")
            with open(os.path.join(temp_dir, "README.md"), "w") as f:
                f.write("# Test Project\n")
            yield temp_dir

    def test_read_file(self, agent, temp_repo):
        result = agent._tool_read_file(temp_repo, {"path": "src/main.py"})
        assert "def greet" in result
        assert "1 |" in result  # Line numbers

    def test_read_file_not_found(self, agent, temp_repo):
        result = agent._tool_read_file(temp_repo, {"path": "nonexistent.py"})
        assert "not found" in result.lower()

    def test_read_file_with_offset(self, agent, temp_repo):
        result = agent._tool_read_file(temp_repo, {"path": "src/main.py", "offset": 2, "limit": 1})
        assert "return" in result
        assert "def greet" not in result

    def test_list_directory(self, agent, temp_repo):
        result = agent._tool_list_directory(temp_repo, {"path": "."})
        assert "src/" in result
        assert "README.md" in result

    def test_list_directory_subdir(self, agent, temp_repo):
        result = agent._tool_list_directory(temp_repo, {"path": "src"})
        assert "main.py" in result

    def test_search_files(self, agent, temp_repo):
        result = agent._tool_search_files(temp_repo, {"pattern": "def greet"})
        assert "src/main.py" in result
        assert "def greet" in result

    def test_search_files_no_match(self, agent, temp_repo):
        result = agent._tool_search_files(temp_repo, {"pattern": "nonexistent_function"})
        assert "No matches" in result

    def test_search_files_with_glob(self, agent, temp_repo):
        result = agent._tool_search_files(temp_repo, {"pattern": "Test", "file_pattern": "*.md"})
        assert "README.md" in result

    def test_edit_file_success(self, agent, temp_repo):
        result = agent._tool_edit_file(
            temp_repo,
            {"path": "src/main.py", "old_text": "Hello, {name}!", "new_text": "Hi, {name}!"},
        )
        assert "Successfully edited" in result
        with open(os.path.join(temp_repo, "src", "main.py")) as f:
            assert "Hi, {name}!" in f.read()

    def test_edit_file_old_text_not_found(self, agent, temp_repo):
        result = agent._tool_edit_file(
            temp_repo,
            {"path": "src/main.py", "old_text": "nonexistent text", "new_text": "replacement"},
        )
        assert "not found" in result.lower()

    def test_edit_file_syntax_error_rejected(self, agent, temp_repo):
        result = agent._tool_edit_file(
            temp_repo,
            {"path": "src/main.py", "old_text": "def greet(name):", "new_text": "def greet(name"},
        )
        assert "syntax error" in result.lower()
        # Verify file was NOT modified
        with open(os.path.join(temp_repo, "src", "main.py")) as f:
            assert "def greet(name):" in f.read()

    def test_edit_file_create_new(self, agent, temp_repo):
        result = agent._tool_edit_file(
            temp_repo,
            {"path": "src/new_file.py", "old_text": "", "new_text": "x = 42\n"},
        )
        assert "created" in result.lower()
        assert os.path.exists(os.path.join(temp_repo, "src", "new_file.py"))

    def test_edit_file_create_existing_fails(self, agent, temp_repo):
        result = agent._tool_edit_file(
            temp_repo,
            {"path": "src/main.py", "old_text": "", "new_text": "overwrite"},
        )
        assert "already exists" in result.lower()

    def test_run_command(self, agent, temp_repo):
        result = agent._tool_run_command(temp_repo, {"command": "echo hello"})
        assert "hello" in result
        assert "Exit code: 0" in result

    def test_run_command_failure(self, agent, temp_repo):
        result = agent._tool_run_command(temp_repo, {"command": "false"})
        assert "Exit code: 1" in result

    def test_safe_path_prevents_traversal(self, agent, temp_repo):
        with pytest.raises(ValueError, match="traversal"):
            agent._safe_path(temp_repo, "../../etc/passwd")

    def test_safe_path_allows_normal(self, agent, temp_repo):
        result = agent._safe_path(temp_repo, "src/main.py")
        assert result.endswith("src/main.py")

    def test_check_syntax_valid(self, agent, temp_repo):
        assert agent._check_syntax(os.path.join(temp_repo, "src", "main.py")) is None

    def test_check_syntax_invalid(self, agent, temp_repo):
        bad_file = os.path.join(temp_repo, "bad.py")
        with open(bad_file, "w") as f:
            f.write("def broken(\n")
        error = agent._check_syntax(bad_file)
        assert error is not None

    def test_detect_new_files(self, agent, temp_repo):
        # Create a new file in the temp repo
        with open(os.path.join(temp_repo, "src", "added.py"), "w") as f:
            f.write("new = True\n")
        original = {"src/main.py": "..."}
        new_files = agent._detect_new_files(temp_repo, original)
        assert "src/added.py" in new_files
        assert new_files["src/added.py"] == "new = True\n"


# --- Repo structure map tests ---


class TestRepoStructureMap:
    def test_build_repo_structure_map(self):
        from darwinian_evolver.problems.repo_task_agent import build_repo_structure_map

        files = {
            "src/math_utils.py": "def add(a, b):\n    return a + b\n\n\nclass Calculator:\n    def multiply(self, x, y):\n        return x * y\n",
            "tests/test_math.py": "def test_add():\n    assert True\n",
        }
        result = build_repo_structure_map(files)
        assert "src/math_utils.py" in result
        assert "add(a, b)" in result
        assert "Calculator" in result
        assert "multiply(self, x, y)" in result
        assert "test_add()" in result
