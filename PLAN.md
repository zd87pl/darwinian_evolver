# Plan: Agentic Repo Evolution Problem Type

## Goal
Add a new `repo_task` problem type that lets you point the evolutionary framework
at any existing git repository, give it a task description and a test command,
and have it evolve the codebase toward passing all tests / achieving the goal.

The key upgrade: mutators become **multi-turn agentic loops** (using the Anthropic
tool-use API) instead of single-shot LLM calls. Each agent can read files, search
code, edit files, and run commands inside a sandboxed temp clone of the repo.

---

## Step 1: Enhance GitBasedOrganism to support new file creation

**File:** `darwinian_evolver/git_based_problem.py`

Currently `build_repo()` asserts every file in `file_contents` already exists in the
repo. We need to also handle creating new files (the agent might add tests, modules,
etc.).

Changes:
- In `build_repo()`, replace the assertion with `os.makedirs` + write for new files
- Add a helper `capture_modified_files(temp_dir)` that diffs the temp dir against the
  base commit and returns an updated `file_contents` dict of everything that changed
- Add a `deleted_files: set[str]` field for tracking file deletions

---

## Step 2: Create the RepoTask organism, evaluator, and failure case

**New file:** `darwinian_evolver/problems/repo_task.py`

### RepoTaskOrganism (extends GitBasedOrganism)
```python
class RepoTaskOrganism(GitBasedOrganism):
    task_description: str  # What the user wants built/fixed
```

### RepoTaskFailureCase
```python
class RepoTaskFailureCase(EvaluationFailureCase):
    test_name: str        # Individual test/check that failed
    error_output: str     # Captured stderr/stdout for that failure
    failure_type: str     # "test_failure", "syntax_error", "lint_error", etc.
```

### RepoTaskEvaluationResult
```python
class RepoTaskEvaluationResult(EvaluationResult):
    raw_output: str       # Full test command output
    num_passed: int
    num_failed: int
    num_total: int
```

### RepoTaskEvaluator
- Takes a configurable `test_command: str` (e.g., `"pytest --tb=short"`, `"npm test"`, `"make test"`)
- Optional `setup_command: str` for dependency installation in the temp clone
- Optional extra scoring commands (lint, type-check) with configurable weights

Flow:
1. `organism.build_repo()` → temp clone
2. Run `setup_command` if provided (e.g., `pip install -e .`)
3. Run `test_command`, capture stdout/stderr
4. Parse output to extract individual test results
5. Score = `num_passed / num_total` (with bonuses for lint/type-check if configured)
6. Build failure cases from individual test failures

Test output parsing: Start with a generic parser (exit code + raw output),
plus a pytest-specific parser that extracts individual test names and tracebacks.
Users can subclass to add jest, go test, etc.

---

## Step 3: Build the Agentic Mutator

**New file:** `darwinian_evolver/problems/repo_task_agent.py`

This is the core new piece. Uses the Anthropic API with tool-use to run a
multi-turn coding agent inside each mutation step.

### Agent Tools
Define these as Anthropic tool-use tool schemas:

1. **`read_file(path)`** — Read a file from the temp repo
2. **`list_directory(path)`** — List files/dirs (like `ls`)
3. **`search_files(pattern, path?)`** — Grep for a regex pattern
4. **`write_file(path, content)`** — Write/overwrite a file (creates dirs if needed)
5. **`run_command(command)`** — Run a shell command in the repo root (with timeout)

### Agent Loop
```python
class AgenticRepoMutator(Mutator[RepoTaskOrganism, RepoTaskFailureCase]):
    def __init__(self, model, max_turns, task_description):
        self.model = model           # e.g., "claude-sonnet-4-20250514"
        self.max_turns = max_turns   # e.g., 25
        self.task_description = task_description

    def mutate(self, organism, failure_cases, learning_log_entries):
        with organism.build_repo() as temp_dir:
            # Build the initial prompt
            system = AGENT_SYSTEM_PROMPT  # Expert coding agent instructions
            user_msg = render_prompt(
                task=self.task_description,
                current_files=organism.file_contents,
                failures=failure_cases,
                learning_log=learning_log_entries,
            )

            messages = [{"role": "user", "content": user_msg}]

            for turn in range(self.max_turns):
                response = client.messages.create(
                    model=self.model,
                    max_tokens=16384,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )

                # If the model is done (no tool use), break
                if response.stop_reason == "end_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    break

                # Execute tool calls against temp_dir
                tool_results = execute_tool_calls(response, temp_dir)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            # Capture all modified/new files from the temp dir
            new_file_contents = capture_changes(temp_dir, organism)

            # Extract change summary from the agent's final message
            summary = extract_change_summary(messages)

        if new_file_contents == organism.file_contents:
            return []  # No meaningful change

        return [RepoTaskOrganism(
            repo_root=organism.repo_root,
            git_hash=organism.git_hash,
            file_contents=new_file_contents,
            task_description=organism.task_description,
            from_change_summary=summary,
        )]
```

### Agent System Prompt
Instructs the agent to:
- Analyze the failure cases provided
- Explore the codebase to understand context (using read_file, search_files)
- Make targeted changes to fix failures (using write_file)
- Run the test command to verify changes work (using run_command)
- Provide a concise summary of what was changed and why
- The prompt will also include learning log entries so the agent can avoid repeating
  previously-failed approaches

### Tool Execution Safety
- `run_command`: timeout of 120s, cwd locked to temp_dir, no network access consideration
- `write_file`: restricted to within temp_dir (path traversal prevention)
- `read_file`: restricted to within temp_dir

---

## Step 4: Register the problem and add CLI support

### Registry update
**File:** `darwinian_evolver/problems/registry.py`

Add `"repo_task": make_repo_task_problem` to `AVAILABLE_PROBLEMS`.

### CLI args
**File:** `darwinian_evolver/__main__.py`

Add repo_task-specific arguments:
- `--repo_root` (required for repo_task) — path to the target git repository
- `--task` (required for repo_task) — natural language description of the goal
- `--test_command` (required for repo_task) — command to evaluate success
- `--setup_command` (optional) — command to run before tests in each temp clone
- `--files` (optional) — explicit list of files to evolve; if omitted, auto-detect
- `--agent_model` (default: `claude-sonnet-4-20250514`) — model for the agentic mutator
- `--agent_max_turns` (default: 25) — max tool-use round-trips per mutation

### Problem factory
```python
def make_repo_task_problem(
    repo_root, task, test_command, setup_command, files, agent_model, agent_max_turns
) -> Problem:
    if not files:
        files = auto_detect_files(repo_root, task, agent_model)

    initial_organism = RepoTaskOrganism.make_initial_organism_from_repo(
        repo_root=repo_root,
        files_to_capture=files,
        task_description=task,
    )

    evaluator = RepoTaskEvaluator(
        test_command=test_command,
        setup_command=setup_command,
    )

    mutator = AgenticRepoMutator(
        model=agent_model,
        max_turns=agent_max_turns,
        task_description=task,
    )

    return Problem(
        initial_organism=initial_organism,
        evaluator=evaluator,
        mutators=[mutator],
    )
```

---

## Step 5: Auto-detect files to evolve (optional but useful)

When `--files` is not provided, make a single LLM call to analyze the task
description + repo structure and identify the most relevant files to capture.

```python
def auto_detect_files(repo_root, task, model) -> list[str]:
    # Get repo file tree (git ls-files)
    tree = subprocess.check_output(["git", "ls-files"], cwd=repo_root)

    # Ask LLM which files are relevant to the task
    response = client.messages.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Given this repo structure:\n{tree}\n\nWhich files are "
                       f"most relevant to this task: {task}\n\nReturn just file paths."
        }],
    )
    return parse_file_list(response)
```

---

## Step 6: Verification support (optional quality boost)

Implement `verify_mutation` on `RepoTaskEvaluator`:
- After an agent produces a mutation, quickly re-run the test command
- If at least one previously-failing test now passes (and no new syntax errors),
  the mutation passes verification
- This avoids expensive full evaluation for mutations that didn't help

---

## Example Usage

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Basic usage: evolve a repo to pass its tests
uv run darwinian_evolver repo_task \
    --repo_root /path/to/my/project \
    --task "Implement the user authentication module with JWT tokens" \
    --test_command "pytest tests/ --tb=short" \
    --num_iterations 10 \
    --num_parents_per_iteration 3 \
    --output_dir /tmp/evolution_output \
    --learning_log neighborhood-2

# With explicit files and setup
uv run darwinian_evolver repo_task \
    --repo_root /path/to/my/project \
    --task "Fix the failing integration tests" \
    --test_command "pytest tests/integration/ -x --tb=long" \
    --setup_command "pip install -e ." \
    --files src/auth.py src/middleware.py src/models/user.py \
    --agent_model claude-sonnet-4-20250514 \
    --agent_max_turns 30 \
    --num_iterations 5 \
    --verify_mutations \
    --output_dir /tmp/fix_output
```

After running, inspect evolution with `lineage_visualizer.html` and extract the
best organism's `file_contents` to apply to your actual repo.

---

## New/Modified Files Summary

| File | Action |
|------|--------|
| `darwinian_evolver/git_based_problem.py` | Modify — support new files + capture helper |
| `darwinian_evolver/problems/repo_task.py` | **Create** — organism, evaluator, failure case, problem factory |
| `darwinian_evolver/problems/repo_task_agent.py` | **Create** — agentic mutator + tool definitions + agent loop |
| `darwinian_evolver/problems/registry.py` | Modify — register repo_task |
| `darwinian_evolver/__main__.py` | Modify — add repo_task CLI args |

No new dependencies needed — the Anthropic SDK (already `anthropic==0.78.0` in
pyproject.toml) supports tool-use natively.
