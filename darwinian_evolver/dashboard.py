"""
Streamlit dashboard for monitoring darwinian_evolver evolution runs.

Launch with:
    uv run --group dashboard streamlit run darwinian_evolver/dashboard.py

Or point at an existing output directory:
    uv run --group dashboard streamlit run darwinian_evolver/dashboard.py -- --output_dir /tmp/evolution_output
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# --- Page config ---
st.set_page_config(page_title="Darwinian Evolver", page_icon="🧬", layout="wide")


# --- Data loading ---


@st.cache_data(ttl=2)
def load_results(results_file: Path) -> list[dict]:
    """Load results from a results.jsonl file."""
    if not results_file.exists():
        return []
    data = []
    for line in results_file.read_text().strip().split("\n"):
        if line.strip():
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def get_organisms_from_iteration(iteration_data: dict) -> list[dict]:
    """Extract organism data from an iteration."""
    population = iteration_data.get("population", {})
    return population.get("organisms", [])


def get_best_organism(iteration_data: dict) -> dict | None:
    """Get the highest-scoring organism from an iteration."""
    organisms = get_organisms_from_iteration(iteration_data)
    if not organisms:
        return None
    return max(organisms, key=lambda o: o.get("evaluation_result", {}).get("score", 0))


# --- Components ---


def render_fitness_chart(data: list[dict]) -> None:
    """Render the fitness over generations chart."""
    if not data:
        st.info("Waiting for first iteration...")
        return

    iterations = []
    best_scores = []
    median_scores = []
    mean_scores = []

    for d in data:
        iterations.append(d["iteration"])
        organisms = get_organisms_from_iteration(d)
        scores = [o.get("evaluation_result", {}).get("score", 0) for o in organisms]
        if scores:
            best_scores.append(max(scores))
            sorted_scores = sorted(scores)
            median_scores.append(sorted_scores[len(sorted_scores) // 2])
            mean_scores.append(sum(scores) / len(scores))
        else:
            best_scores.append(0)
            median_scores.append(0)
            mean_scores.append(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=best_scores, name="Best", line=dict(color="#2ecc71", width=3)))
    fig.add_trace(
        go.Scatter(x=iterations, y=median_scores, name="Median", line=dict(color="#3498db", width=2, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=mean_scores, name="Mean", line=dict(color="#95a5a6", width=1, dash="dot"))
    )
    fig.update_layout(
        title="Fitness Over Generations",
        xaxis_title="Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=400,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_population_stats(data: list[dict]) -> None:
    """Render population statistics."""
    if not data:
        return

    latest = data[-1]
    organisms = get_organisms_from_iteration(latest)
    scores = [o.get("evaluation_result", {}).get("score", 0) for o in organisms]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Iteration", latest["iteration"])
    with col2:
        st.metric("Population Size", len(organisms))
    with col3:
        st.metric("Best Score", f"{max(scores):.4f}" if scores else "N/A")
    with col4:
        if len(data) > 1:
            prev_scores = [
                o.get("evaluation_result", {}).get("score", 0) for o in get_organisms_from_iteration(data[-2])
            ]
            prev_best = max(prev_scores) if prev_scores else 0
            curr_best = max(scores) if scores else 0
            delta = curr_best - prev_best
            st.metric("Best Score Delta", f"{delta:+.4f}")
        else:
            st.metric("Best Score Delta", "N/A")


def render_best_organism(data: list[dict]) -> None:
    """Show the best organism's code and details."""
    if not data:
        st.info("No data yet.")
        return

    latest = data[-1]
    best = get_best_organism(latest)
    if best is None:
        st.warning("No organisms found.")
        return

    organism = best.get("organism", {})
    result = best.get("evaluation_result", {})

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Score", f"{result.get('score', 0):.4f}")
        if "num_passed" in result:
            st.metric("Tests Passed", f"{result['num_passed']}/{result.get('num_total', '?')}")
        if organism.get("from_change_summary"):
            st.markdown("**Change summary:**")
            st.caption(organism["from_change_summary"][:300])

    with col2:
        file_contents = organism.get("file_contents", {})
        if file_contents:
            tabs = st.tabs(list(file_contents.keys()))
            for tab, (path, content) in zip(tabs, file_contents.items()):
                with tab:
                    lang = _detect_language(path)
                    st.code(content, language=lang, line_numbers=True)


def render_diff_view(data: list[dict]) -> None:
    """Show diffs between iterations."""
    if len(data) < 2:
        st.info("Need at least 2 iterations to show diffs.")
        return

    iteration_options = list(range(len(data)))
    selected = st.selectbox("Compare iteration", iteration_options[1:], index=len(iteration_options) - 2)

    curr_best = get_best_organism(data[selected])
    prev_best = get_best_organism(data[selected - 1])

    if not curr_best or not prev_best:
        st.warning("Could not find organisms to compare.")
        return

    curr_files = curr_best.get("organism", {}).get("file_contents", {})
    prev_files = prev_best.get("organism", {}).get("file_contents", {})

    all_files = sorted(set(curr_files.keys()) | set(prev_files.keys()))
    changed_files = [f for f in all_files if curr_files.get(f, "") != prev_files.get(f, "")]

    if not changed_files:
        st.success("No changes between these iterations.")
        return

    for file_path in changed_files:
        old_content = prev_files.get(file_path, "")
        new_content = curr_files.get(file_path, "")
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"iter {selected - 1}: {file_path}",
            tofile=f"iter {selected}: {file_path}",
        )
        diff_str = "".join(diff)
        if diff_str:
            with st.expander(f"{file_path}", expanded=True):
                st.code(diff_str, language="diff")


def render_mutation_log(data: list[dict]) -> None:
    """Show the mutation log across iterations."""
    if not data:
        st.info("No data yet.")
        return

    entries = []
    for d in data:
        iteration = d["iteration"]
        organisms = get_organisms_from_iteration(d)
        for org_data in organisms:
            organism = org_data.get("organism", {})
            result = org_data.get("evaluation_result", {})
            summary = organism.get("from_change_summary")
            if summary:
                entries.append(
                    {
                        "iteration": iteration,
                        "score": result.get("score", 0),
                        "change": summary[:200],
                        "parent_id": organism.get("parent_id", "root")[:8] if organism.get("parent_id") else "root",
                        "id": organism.get("id", "?")[:8],
                    }
                )

    if not entries:
        st.info("No mutations recorded yet.")
        return

    # Show most recent first
    for entry in reversed(entries[-50:]):
        score_color = "green" if entry["score"] > 0.7 else "orange" if entry["score"] > 0.3 else "red"
        st.markdown(
            f"**Iter {entry['iteration']}** | "
            f":{score_color}[{entry['score']:.2f}] | "
            f"`{entry['parent_id']}` -> `{entry['id']}` | "
            f"{entry['change']}"
        )


def render_verification_failures(data: list[dict]) -> None:
    """Show organisms that failed verification."""
    if not data:
        return

    latest = data[-1]
    failures = latest.get("verification_failures", [])
    if not failures:
        st.success("No verification failures in latest iteration.")
        return

    st.warning(f"{len(failures)} organism(s) failed verification")
    for org in failures[:10]:
        summary = org.get("from_change_summary", "No summary")
        st.caption(f"- {summary[:150]}")


# --- Run Manager ---


class RunManager:
    """Manages evolution subprocess lifecycle via Streamlit session state."""

    @staticmethod
    def start_run(config: dict) -> subprocess.Popen:
        cmd = [sys.executable, "-m", "darwinian_evolver", config["problem"]]
        for key, value in config.items():
            if key == "problem" or value is None or value == "" or value is False:
                continue
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.append(f"--{key}")
                cmd.extend(str(v) for v in value)
            else:
                cmd.extend([f"--{key}", str(value)])

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return process

    @staticmethod
    def stop_run(process: subprocess.Popen) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    @staticmethod
    def is_running(process: subprocess.Popen | None) -> bool:
        if process is None:
            return False
        return process.poll() is None


# --- Sidebar ---


def render_sidebar() -> Path | None:
    """Render the sidebar with run configuration and return the results file path."""
    with st.sidebar:
        st.title("Darwinian Evolver")

        mode = st.radio("Mode", ["Watch existing run", "Launch new run"], horizontal=True)

        if mode == "Watch existing run":
            output_dir = st.text_input("Output directory", value=st.session_state.get("output_dir", ""))
            if output_dir:
                st.session_state["output_dir"] = output_dir
                results_file = Path(output_dir) / "results.jsonl"
                if results_file.exists():
                    st.success("Found results.jsonl")
                else:
                    st.warning("results.jsonl not found yet")
                return results_file
            return None

        else:
            st.subheader("Configuration")
            problem = st.selectbox("Problem type", ["repo_task", "spec_task"])

            repo_root = st.text_input("Repository path")
            output_dir = st.text_input("Output directory", value="/tmp/evolution_output")

            if problem == "repo_task":
                task = st.text_area("Task description", height=100)
                test_command = st.text_input("Test command", value="pytest --tb=short")
                setup_command = st.text_input("Setup command (optional)")
            else:
                task = st.text_area("Task description (optional)", height=60)
                spec = st.text_area("Specification", height=150)
                validation_command = st.text_input("Validation command (optional)")

            st.subheader("Parameters")
            agent_model = st.selectbox("Agent model", ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"])
            num_iterations = st.slider("Iterations", 1, 50, 10)
            num_parents = st.slider("Parents per iteration", 1, 10, 4)
            max_turns = st.slider("Max agent turns", 5, 50, 25)
            learning_log = st.selectbox("Learning log", ["none", "ancestors", "neighborhood-2", "neighborhood-3"])

            col1, col2 = st.columns(2)
            with col1:
                start = st.button("Start", type="primary", use_container_width=True)
            with col2:
                stop = st.button("Stop", type="secondary", use_container_width=True)

            if stop and "process" in st.session_state:
                RunManager.stop_run(st.session_state["process"])
                del st.session_state["process"]
                st.success("Run stopped.")

            if start:
                config = {
                    "problem": problem,
                    "repo_root": repo_root,
                    "output_dir": output_dir,
                    "agent_model": agent_model,
                    "num_iterations": num_iterations,
                    "num_parents_per_iteration": num_parents,
                    "agent_max_turns": max_turns,
                    "learning_log": learning_log,
                }
                if problem == "repo_task":
                    config["task"] = task
                    config["test_command"] = test_command
                    if setup_command:
                        config["setup_command"] = setup_command
                else:
                    if task:
                        config["task"] = task
                    config["spec"] = spec
                    if validation_command:
                        config["validation_command"] = validation_command

                process = RunManager.start_run(config)
                st.session_state["process"] = process
                st.session_state["output_dir"] = output_dir
                st.success("Run started!")

            # Show run status
            if "process" in st.session_state:
                if RunManager.is_running(st.session_state["process"]):
                    st.info("Evolution is running...")
                else:
                    st.warning("Process has finished.")

            if output_dir:
                return Path(output_dir) / "results.jsonl"
            return None


# --- Helpers ---


def _detect_language(path: str) -> str:
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return "text"


# --- Main ---


def main() -> None:
    results_file = render_sidebar()

    if results_file is None:
        st.header("Welcome to Darwinian Evolver")
        st.markdown(
            "Configure a run in the sidebar, or point to an existing output directory to watch a running evolution."
        )
        st.markdown(
            """
### Quick start

**Fix failing tests in an existing repo:**
```bash
uv run darwinian_evolver repo_task \\
    --repo_root /path/to/project \\
    --task "Fix the authentication bug" \\
    --test_command "pytest tests/ --tb=short" \\
    --output_dir /tmp/evolution_output
```

**Build something from a specification:**
```bash
uv run darwinian_evolver spec_task \\
    --repo_root /path/to/project \\
    --spec "Implement a REST API with endpoints for CRUD operations on users" \\
    --validation_command "python -c 'import app'" \\
    --output_dir /tmp/evolution_output
```
"""
        )
        return

    data = load_results(results_file)

    # Stats bar
    render_population_stats(data)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fitness", "Best Organism", "Diffs", "Mutation Log", "Verification"])

    with tab1:
        render_fitness_chart(data)
    with tab2:
        render_best_organism(data)
    with tab3:
        render_diff_view(data)
    with tab4:
        render_mutation_log(data)
    with tab5:
        render_verification_failures(data)

    # Auto-refresh
    refresh_interval = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)
    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    # Parse --output_dir from command line if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.output_dir:
        st.session_state["output_dir"] = args.output_dir
        st.session_state["_watch_mode"] = True

    main()
