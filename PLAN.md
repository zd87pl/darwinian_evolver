# Plan: Agentic Repo Evolution + Streamlit Dashboard

## Overview

Two major additions to darwinian_evolver:

1. **`repo_task` problem type** — Point the evolutionary framework at any git repository
   with a task description and test command. Mutators become multi-turn coding agents
   (using Anthropic tool-use API) that can read, write, search, and run tests.

2. **Streamlit dashboard** — Live web UI for launching, monitoring, and inspecting
   evolution runs. Real-time fitness charts, population trees, code diffs, and
   start/stop controls.

Optional: An MCP server wrapper for ruflo/Claude Code integration (Phase 3).

---

## Architecture Decisions

### Why Streamlit for the GUI?

After researching Streamlit, Gradio, Panel, FastAPI+React, and Textual:

| Feature | Streamlit | Gradio | Panel | FastAPI+React | Textual |
|---------|-----------|--------|-------|---------------|---------|
| Live-updating charts | `st.empty()` + rerun loop | Limited (polling) | Good (param watch) | Full control | No charts |
| Start/Stop/Pause | subprocess management | Limited | Possible | Full control | Good |
| Code diffs display | `st.code()` + difflib | Limited | Limited | Full control | Limited |
| Python-native | Yes | Yes | Yes | No (needs JS) | Yes |
| Integration complexity | Low (~1 file) | Low | Medium | High | Low |
| Multiple concurrent runs | Via session state | Difficult | Possible | Natural | Difficult |

**Streamlit wins** because it's Python-native, has excellent real-time charting
(Plotly integration), good code display, and the `st.empty()` pattern works
perfectly with our `results.jsonl` file-watching model. It also has session state
for managing run lifecycles.

### Why multi-turn tool-use for the mutator?

State-of-the-art coding agents (SWE-agent, aider, OpenHands, Claude Code) all
converge on the same pattern:

1. **ReAct-style loop** — LLM reasons, picks a tool, observes result, repeats
2. **Constrained tool set** — file read/write, search, bash execution
3. **Test feedback** — agent runs tests and iterates within a single turn
4. **Context management** — keep conversation focused, prune when needed

The Anthropic tool-use API (`tools` parameter + `tool_use`/`tool_result` message
types) implements this natively. No framework needed — just a while loop.

Key design choices from SWE-bench research:
- **Fewer, sharper tools** outperform many fine-grained tools
- **Let the agent self-verify** by running tests within its turn
- **Include failure context** (test output, error traces) in the initial prompt
- **Learning log** prevents repeating failed approaches (unique to darwinian_evolver)

### Relationship to ruflo

ruflo provides coordination scaffolding (message bus, task queue, MCP server) but
**its agents are in-memory TypeScript objects, not real coding agents**. The concrete
value for our use case is:

- **MCP server** — expose evolution runs as tools accessible from Claude Code
- **Not useful now** — swarm coordination, consensus, topology (overkill for our model)

We'll build the MCP wrapper as an optional Phase 3 after the core is working.

---

## Phase 1: `repo_task` Problem Type

### Step 1.1: Enhance GitBasedOrganism

**File:** `darwinian_evolver/git_based_problem.py`

Currently `build_repo()` asserts every file in `file_contents` already exists.
The agent might need to create new files.

Changes:
- In `build_repo()`, replace assertion with `os.makedirs` + write for new files
- Add `capture_modified_files(temp_dir, git_hash) -> dict[str, str]` static method
  that diffs the temp dir against the base commit, returning updated file_contents
- Add `deleted_files: set[str] = set()` field for tracking file deletions

### Step 1.2: Define RepoTask domain classes

**New file:** `darwinian_evolver/problems/repo_task.py`

```python
class RepoTaskOrganism(GitBasedOrganism):
    task_description: str

class RepoTaskFailureCase(EvaluationFailureCase):
    test_name: str
    error_output: str
    # failure_type: "test_failure" | "syntax_error" | "runtime_error" | "lint_error"

class RepoTaskEvaluationResult(EvaluationResult):
    raw_output: str
    num_passed: int
    num_failed: int
    num_total: int
```

### Step 1.3: Implement RepoTaskEvaluator

**In:** `darwinian_evolver/problems/repo_task.py`

```python
class RepoTaskEvaluator(Evaluator):
    def __init__(self, test_command, setup_command=None, timeout=300):
        ...

    def evaluate(self, organism):
        with organism.build_repo() as temp_dir:
            if self.setup_command:
                run_with_timeout(self.setup_command, temp_dir, timeout=120)
            result = run_with_timeout(self.test_command, temp_dir, timeout=self.timeout)
            failures = self._parse_test_output(result)
            score = (result.num_total - len(failures)) / max(result.num_total, 1)
            return RepoTaskEvaluationResult(
                score=score,
                trainable_failure_cases=failures,
                raw_output=result.stdout + result.stderr,
                num_passed=result.num_total - len(failures),
                num_failed=len(failures),
                num_total=result.num_total,
            )

    def verify_mutation(self, organism):
        # Quick re-run: check if at least one previously-failing test now passes
        ...

    def _parse_test_output(self, result):
        # Generic parser: each line matching "FAIL" / "ERROR" becomes a failure case
        # + pytest-specific parser extracting individual test names and tracebacks
        ...
```

Test output parsing strategy:
1. **Generic** (default): Exit code 0 = all pass, nonzero = parse output for FAIL/ERROR lines
2. **pytest**: Parse `--tb=short` output, extract `test_name::method FAILED` + traceback
3. **Extensible**: Users subclass evaluator to add jest, go test, cargo test, etc.

### Step 1.4: Build the Agentic Mutator

**New file:** `darwinian_evolver/problems/repo_task_agent.py`

This is the core new piece. A multi-turn tool-use agent loop.

#### Tool Definitions (Anthropic tool-use format)

```python
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path from repo root"}
        }, "required": ["path"]}
    },
    {
        "name": "list_directory",
        "description": "List files and directories",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path, defaults to '.'"}
        }}
    },
    {
        "name": "search_files",
        "description": "Search for a regex pattern across files",
        "input_schema": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search in"},
            "file_pattern": {"type": "string", "description": "Glob pattern e.g. '*.py'"}
        }, "required": ["pattern"]}
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        }, "required": ["path", "content"]}
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the repo root",
        "input_schema": {"type": "object", "properties": {
            "command": {"type": "string"}
        }, "required": ["command"]}
    }
]
```

Design rationale: 5 tools total, matching the SWE-agent "ACI" pattern that
achieves best results on SWE-bench. More tools = more confusion for the model.

#### Agent Loop

```python
class AgenticRepoMutator(Mutator[RepoTaskOrganism, RepoTaskFailureCase]):
    def __init__(self, model="claude-sonnet-4-20250514", max_turns=25):
        ...

    def mutate(self, organism, failure_cases, learning_log_entries):
        with organism.build_repo() as temp_dir:
            messages = [{"role": "user", "content": self._build_prompt(
                organism, failure_cases, learning_log_entries
            )}]

            for turn in range(self.max_turns):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=16384,
                    system=AGENT_SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    break

                # Execute tool calls, build tool_result messages
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block, temp_dir)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})

            # Capture all changes from the temp dir
            new_file_contents = GitBasedOrganism.capture_modified_files(
                temp_dir, organism.git_hash
            )
            summary = self._extract_summary(messages)

        if new_file_contents == organism.file_contents:
            return []  # Agent made no changes

        return [RepoTaskOrganism(
            repo_root=organism.repo_root,
            git_hash=organism.git_hash,
            file_contents=new_file_contents,
            task_description=organism.task_description,
            from_change_summary=summary,
        )]
```

#### Agent System Prompt

Key elements:
- "You are an expert software engineer. Your goal is to modify code to fix failing tests."
- Instruction to explore before editing (read relevant files first)
- Instruction to run tests after making changes to verify
- Warning not to make unnecessary changes or introduce new failures
- Format for the final summary (concise, specific)

#### Tool Execution Safety

```python
def _execute_tool(self, tool_block, temp_dir):
    name, args = tool_block.name, tool_block.input

    if name == "read_file":
        path = self._safe_path(temp_dir, args["path"])
        return open(path).read()

    elif name == "write_file":
        path = self._safe_path(temp_dir, args["path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(args["content"])
        return "File written successfully."

    elif name == "run_command":
        result = subprocess.run(
            args["command"], shell=True, cwd=temp_dir,
            capture_output=True, text=True, timeout=120
        )
        return f"Exit code: {result.returncode}\n{result.stdout}\n{result.stderr}"
    ...

def _safe_path(self, temp_dir, relative_path):
    """Prevent path traversal attacks."""
    resolved = os.path.realpath(os.path.join(temp_dir, relative_path))
    if not resolved.startswith(os.path.realpath(temp_dir)):
        raise ValueError(f"Path traversal detected: {relative_path}")
    return resolved
```

### Step 1.5: Register and wire up CLI

**File:** `darwinian_evolver/problems/registry.py`
- Add `"repo_task": make_repo_task_problem`
- The factory needs extra args, so we'll use a partial/lambda from CLI args

**File:** `darwinian_evolver/__main__.py`
- Add repo_task-specific argument group:
  - `--repo_root` — path to target git repository
  - `--task` — natural language task description
  - `--test_command` — command to evaluate (e.g., `"pytest --tb=short"`)
  - `--setup_command` — optional setup command
  - `--files` — optional explicit file list (auto-detect if omitted)
  - `--agent_model` — model for agentic mutator (default: claude-sonnet-4-20250514)
  - `--agent_max_turns` — max tool-use turns (default: 25)

### Step 1.6: Auto-detect files to evolve

When `--files` is omitted, make a single LLM call analyzing the task + repo tree
(`git ls-files`) to identify relevant files. This prevents capturing the entire
repo in every organism.

---

## Phase 2: Streamlit Dashboard

### Step 2.1: Dashboard Architecture

**New file:** `darwinian_evolver/dashboard.py` (single file, ~400-600 lines)

The dashboard runs as a separate process alongside evolution runs:
```bash
# Terminal 1: Run evolution
uv run darwinian_evolver repo_task --output_dir /tmp/output ...

# Terminal 2: Watch it live
uv run streamlit run darwinian_evolver/dashboard.py
```

OR launch everything from the dashboard itself (preferred UX).

Architecture:
```
┌──────────────────────────────────────────────────┐
│                 Streamlit Dashboard               │
│                                                   │
│  ┌─────────────┐  ┌──────────────────────────┐   │
│  │  Sidebar     │  │  Main Area                │  │
│  │             │  │                            │  │
│  │ Run Config  │  │  Tab 1: Live Fitness Chart │  │
│  │  - problem  │  │    (Plotly line chart)     │  │
│  │  - repo     │  │                            │  │
│  │  - task     │  │  Tab 2: Population Tree    │  │
│  │  - tests    │  │    (D3 lineage or Plotly)  │  │
│  │  - params   │  │                            │  │
│  │             │  │  Tab 3: Best Organism      │  │
│  │ [Start]     │  │    (code + diff + score)   │  │
│  │ [Stop]      │  │                            │  │
│  │ [Pause]     │  │  Tab 4: Mutation Log       │  │
│  │             │  │    (learning log entries)   │  │
│  │ Run History │  │                            │  │
│  │  - run_1    │  │  Tab 5: Stats              │  │
│  │  - run_2    │  │    (evolver stats, timing)  │  │
│  └─────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Step 2.2: Core Dashboard Components

#### a) Run Launcher (Sidebar)

```python
with st.sidebar:
    st.header("Evolution Run")
    problem = st.selectbox("Problem", list(AVAILABLE_PROBLEMS.keys()))

    if problem == "repo_task":
        repo_root = st.text_input("Repository Path")
        task = st.text_area("Task Description")
        test_command = st.text_input("Test Command", "pytest --tb=short")
        setup_command = st.text_input("Setup Command (optional)")
        agent_model = st.selectbox("Agent Model", [
            "claude-sonnet-4-20250514", "claude-opus-4-6"
        ])
        max_turns = st.slider("Max Agent Turns", 5, 50, 25)

    num_iterations = st.slider("Iterations", 1, 100, 10)
    num_parents = st.slider("Parents per Iteration", 1, 10, 4)
    output_dir = st.text_input("Output Directory", "/tmp/evolution_output")

    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Start", type="primary")
    with col2:
        stop = st.button("Stop", type="secondary")
```

#### b) Live Fitness Chart

```python
def render_fitness_chart(results_file):
    """Read results.jsonl and plot fitness over iterations."""
    if not results_file.exists():
        st.info("Waiting for first iteration...")
        return

    data = [json.loads(line) for line in results_file.read_text().strip().split('\n')]

    iterations = [d["iteration"] for d in data]
    best_scores = []
    median_scores = []

    for d in data:
        organisms = d["population"]["organisms"]
        scores = [o["evaluation_result"]["score"] for o in organisms]
        best_scores.append(max(scores))
        median_scores.append(sorted(scores)[len(scores)//2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=best_scores, name="Best"))
    fig.add_trace(go.Scatter(x=iterations, y=median_scores, name="Median"))
    fig.update_layout(title="Fitness Over Generations",
                      xaxis_title="Iteration", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
```

#### c) Best Organism Viewer

```python
def render_best_organism(results_file):
    """Show the best organism's code and diff from parent."""
    data = load_latest_results(results_file)
    best = max(data["population"]["organisms"],
               key=lambda o: o["evaluation_result"]["score"])

    organism = best["organism"]
    result = best["evaluation_result"]

    st.metric("Best Score", f"{result['score']:.4f}")

    if "file_contents" in organism:
        for path, content in organism["file_contents"].items():
            with st.expander(f"📄 {path}"):
                st.code(content, language=_detect_language(path))

    if "diff_from_parent" in organism and organism["diff_from_parent"]:
        with st.expander("📝 Diff from Parent"):
            st.code(organism["diff_from_parent"], language="diff")
```

#### d) Auto-Refresh Loop

```python
# Main dashboard loop with auto-refresh
placeholder = st.empty()
refresh_interval = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)

while True:
    with placeholder.container():
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Fitness", "Population", "Best Organism", "Mutation Log"]
        )
        with tab1:
            render_fitness_chart(results_file)
        with tab2:
            render_population_tree(results_file)
        with tab3:
            render_best_organism(results_file)
        with tab4:
            render_mutation_log(results_file)

    time.sleep(refresh_interval)
    st.rerun()
```

### Step 2.3: Process Management

The dashboard manages evolution runs via subprocess:

```python
class RunManager:
    """Manages evolution subprocess lifecycle."""

    def start_run(self, config: dict) -> subprocess.Popen:
        cmd = ["uv", "run", "darwinian_evolver", config["problem"]]
        for key, value in config.items():
            if key != "problem" and value:
                cmd.extend([f"--{key}", str(value)])
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process

    def stop_run(self, process: subprocess.Popen):
        process.terminate()
        process.wait(timeout=10)

    def is_running(self, process: subprocess.Popen) -> bool:
        return process.poll() is None
```

Store active processes in `st.session_state` for persistence across reruns.

### Step 2.4: Dependencies

Add to `pyproject.toml` as optional dependency group:

```toml
[dependency-groups]
dashboard = [
    "streamlit>=1.40.0",
    "plotly>=5.24.0",
    "watchdog>=6.0.0",  # For file watching (optional optimization)
]
```

Run with: `uv run --group dashboard streamlit run darwinian_evolver/dashboard.py`

---

## Phase 3: ruflo / MCP Integration (Optional, Future)

### Step 3.1: MCP Server Wrapper

**New file:** `darwinian_evolver/mcp_server.py`

Expose evolution runs as MCP tools using the Python `mcp` package:

```python
@mcp.tool()
def start_evolution(repo_root: str, task: str, test_command: str, ...):
    """Start an evolution run and return a run ID."""

@mcp.tool()
def get_evolution_status(run_id: str):
    """Get current status: iteration, best score, population size."""

@mcp.tool()
def get_best_organism(run_id: str):
    """Get the best organism's code and diff."""

@mcp.tool()
def stop_evolution(run_id: str):
    """Stop a running evolution."""
```

This would let ruflo's CLI or any MCP-compatible client (Claude Code, VS Code,
Cursor) drive evolution runs.

### Step 3.2: ruflo Bridge (if desired)

Install ruflo (`npx ruflo@alpha`) and configure it to call our MCP server.
This gives access to ruflo's task management UI and multi-run coordination.

---

## New/Modified Files Summary

| File | Action | Phase |
|------|--------|-------|
| `darwinian_evolver/git_based_problem.py` | Modify — new file creation + capture helper | 1 |
| `darwinian_evolver/problems/repo_task.py` | **Create** — organism, evaluator, failure case | 1 |
| `darwinian_evolver/problems/repo_task_agent.py` | **Create** — agentic mutator + tools + loop | 1 |
| `darwinian_evolver/problems/registry.py` | Modify — register repo_task | 1 |
| `darwinian_evolver/__main__.py` | Modify — add repo_task CLI args | 1 |
| `darwinian_evolver/dashboard.py` | **Create** — Streamlit dashboard | 2 |
| `pyproject.toml` | Modify — add dashboard dependency group | 2 |
| `darwinian_evolver/mcp_server.py` | **Create** — MCP tool server | 3 |

## Dependencies

- **Phase 1**: No new deps (anthropic SDK already supports tool-use)
- **Phase 2**: `streamlit`, `plotly` (optional dashboard group)
- **Phase 3**: `mcp` Python package (optional)

## Implementation Order

1. Step 1.1 (GitBasedOrganism enhancement) — foundation
2. Steps 1.2-1.3 (domain classes + evaluator) — can test with manual mutations
3. Step 1.4 (agentic mutator) — the core value-add
4. Steps 1.5-1.6 (CLI + auto-detect) — make it usable
5. Step 2.1-2.4 (Streamlit dashboard) — make it visual
6. Phase 3 (MCP) — optional, when/if needed

## Example Usage

```bash
# Phase 1: Run evolution from CLI
uv run darwinian_evolver repo_task \
    --repo_root /path/to/my/project \
    --task "Implement JWT authentication for the /api/auth endpoints" \
    --test_command "pytest tests/ --tb=short" \
    --setup_command "pip install -e ." \
    --num_iterations 10 \
    --num_parents_per_iteration 3 \
    --learning_log neighborhood-2 \
    --output_dir /tmp/evolution_output

# Phase 2: Launch dashboard
uv run --group dashboard streamlit run darwinian_evolver/dashboard.py

# Phase 2 alternative: Launch everything from the dashboard UI
uv run --group dashboard streamlit run darwinian_evolver/dashboard.py
# Then configure and start the run from the web UI
```
