"""
Multi-turn agentic mutator for the repo_task problem type.

Uses the Anthropic tool-use API to drive a coding agent that can explore the repo,
make edits, and run tests within a single mutation turn.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess

from anthropic import Anthropic

from darwinian_evolver.git_based_problem import GitBasedOrganism
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problems.repo_task import RepoTaskFailureCase
from darwinian_evolver.problems.repo_task import RepoTaskOrganism


# --- Repo Structure Map ---


def build_repo_structure_map(file_contents: dict[str, str]) -> str:
    """
    Build a concise structural map of the repository from file contents.

    For Python files, parses the AST to extract function/class/method signatures.
    For other files, just lists the file path.
    """
    lines = []
    for path in sorted(file_contents.keys()):
        content = file_contents[path]
        if path.endswith(".py"):
            signatures = _extract_python_signatures(content)
            if signatures:
                lines.append(f"{path}:")
                for sig in signatures:
                    lines.append(f"  {sig}")
            else:
                lines.append(f"{path}: (empty or unparseable)")
        else:
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            lines.append(f"{path}: ({line_count} lines)")
    return "\n".join(lines)


def _extract_python_signatures(source: str) -> list[str]:
    """Extract function and class signatures from Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    signatures = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = _format_args(node.args)
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            signatures.append(f"{prefix}{node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            class_sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            signatures.append(class_sig)
            # Extract methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    args = _format_args(item.args)
                    prefix = "async def " if isinstance(item, ast.AsyncFunctionDef) else "def "
                    signatures.append(f"  {prefix}{item.name}({args})")

    return signatures


def _format_args(args: ast.arguments) -> str:
    """Format function arguments into a concise signature string."""
    parts = []
    # Regular args (skip 'self' and 'cls')
    for arg in args.args:
        if arg.arg not in ("self", "cls"):
            parts.append(arg.arg)
        else:
            parts.append(arg.arg)
    return ", ".join(parts)


# --- Tool Definitions ---

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Returns the file content with line numbers. "
            "For large files, use the offset and limit parameters to read specific sections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from repo root",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed). Defaults to 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Defaults to 200.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and directories at the given path. Returns names with '/' suffix for directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from repo root. Defaults to '.' (repo root).",
                },
            },
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a regex pattern across files in the repository. "
            "Returns matching lines with file paths and line numbers. "
            "Results are capped at 50 matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (relative to repo root). Defaults to '.'.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files, e.g. '*.py'. Defaults to all files.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit a file by replacing an exact string match with new content. "
            "The old_text must match exactly (including whitespace and indentation). "
            "If old_text is empty and the file does not exist, creates a new file with new_text. "
            "The edit will be rejected if it introduces a syntax error in Python files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from repo root",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace. Empty string to create a new file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace old_text with, or full content for new files.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the repo root directory. "
            "Use this to run tests, install dependencies, or inspect the environment. "
            "Output is capped at 3000 characters. Timeout: 120 seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
]


# --- System Prompt ---

AGENT_SYSTEM_PROMPT = """You are an expert software engineer working on a coding task in a git repository.

Your goal is to modify the code to fix failing tests or implement the requested changes.

## Approach

1. **Explore first**: Read the relevant files and understand the codebase before making changes.
2. **Make targeted changes**: Only modify what's necessary. Avoid unrelated refactors.
3. **Verify your work**: Run the test command after making changes to confirm they work.
4. **Iterate if needed**: If tests still fail, read the error output carefully and try again.

## Rules

- Do NOT make unnecessary changes or "improve" code that isn't related to the task.
- Do NOT introduce new dependencies unless absolutely required.
- Do NOT delete or comment out tests to make them pass.
- Keep your changes minimal and focused.
- If you're unsure about something, read the relevant code first.

## When you're done

After you've made your changes and verified they work (or made your best attempt),
stop and provide a brief summary of what you changed and why.
"""


# --- Agent Mutator ---


class AgenticRepoMutator(Mutator[RepoTaskOrganism, RepoTaskFailureCase]):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 25,
    ) -> None:
        super().__init__()
        self._model = model
        self._max_turns = max_turns
        self._client = Anthropic()
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def mutate(
        self,
        organism: RepoTaskOrganism,
        failure_cases: list[RepoTaskFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[RepoTaskOrganism]:
        mutation_input_tokens = 0
        mutation_output_tokens = 0

        with organism.build_repo() as temp_dir:
            user_prompt = self._build_prompt(organism, failure_cases, learning_log_entries)
            messages: list[dict] = [{"role": "user", "content": user_prompt}]

            for _turn in range(self._max_turns):
                try:
                    response = self._client.messages.create(
                        model=self._model,
                        max_tokens=16384,
                        system=AGENT_SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=messages,
                    )
                except Exception as e:
                    print(f"[AgenticMutator] API error: {e}")
                    break

                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    mutation_input_tokens += response.usage.input_tokens
                    mutation_output_tokens += response.usage.output_tokens

                # Append the assistant's response
                messages.append({"role": "assistant", "content": response.content})

                # If the model stopped without requesting tools, we're done
                if response.stop_reason == "end_turn":
                    break

                # Execute tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input, temp_dir)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                if not tool_results:
                    break

                messages.append({"role": "user", "content": tool_results})

            # Capture changes
            new_file_contents = GitBasedOrganism.capture_modified_files(temp_dir, organism.file_contents)

            # Detect newly created files (files in temp_dir that weren't in original)
            new_files = self._detect_new_files(temp_dir, organism.file_contents)
            new_file_contents.update(new_files)

            summary = self._extract_summary(messages)

        # Update cumulative cost tracking
        self.total_input_tokens += mutation_input_tokens
        self.total_output_tokens += mutation_output_tokens
        self.total_api_calls += 1

        # If nothing changed, return empty
        if new_file_contents == organism.file_contents and not new_files:
            return []

        return [
            RepoTaskOrganism(
                repo_root=organism.repo_root,
                git_hash=organism.git_hash,
                file_contents=new_file_contents,
                task_description=organism.task_description,
                from_change_summary=summary,
            )
        ]

    def _build_prompt(
        self,
        organism: RepoTaskOrganism,
        failure_cases: list[RepoTaskFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> str:
        parts = []

        # Task description
        parts.append(f"## Task\n\n{organism.task_description}")

        # Repo structure map (AST-based signatures)
        structure_map = build_repo_structure_map(organism.file_contents)
        parts.append(f"## Repository structure\n\n```\n{structure_map}\n```")

        # Failure cases
        if failure_cases:
            parts.append("## Failing tests\n\nThe following tests are currently failing:")
            for fc in failure_cases:
                parts.append(f"### {fc.test_name}")
                if fc.error_output:
                    # Truncate long error output
                    error = fc.error_output[:1500]
                    parts.append(f"```\n{error}\n```")

        # Learning log
        if learning_log_entries:
            parts.append("## Previous attempts (learning log)\n\nLearn from these previous attempts:")
            for entry in learning_log_entries[-5:]:  # Last 5 entries
                parts.append(f"- **Attempted:** {entry.attempted_change}")
                parts.append(f"  **Outcome:** {entry.observed_outcome}")

        # Instructions
        parts.append(
            "## Instructions\n\n"
            "1. Start by reading the failing test files and the source code they test.\n"
            "2. Understand what the tests expect.\n"
            "3. Make the minimal changes needed to fix the failing tests.\n"
            "4. Run the test command to verify your changes work.\n"
            "5. When done, provide a brief summary of your changes."
        )

        return "\n\n".join(parts)

    def _execute_tool(self, name: str, args: dict, temp_dir: str) -> str:
        try:
            if name == "read_file":
                return self._tool_read_file(temp_dir, args)
            elif name == "list_directory":
                return self._tool_list_directory(temp_dir, args)
            elif name == "search_files":
                return self._tool_search_files(temp_dir, args)
            elif name == "edit_file":
                return self._tool_edit_file(temp_dir, args)
            elif name == "run_command":
                return self._tool_run_command(temp_dir, args)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {e}"

    def _tool_read_file(self, temp_dir: str, args: dict) -> str:
        path = self._safe_path(temp_dir, args["path"])
        if not os.path.exists(path):
            return f"File not found: {args['path']}"
        if not os.path.isfile(path):
            return f"Not a file: {args['path']}"

        with open(path) as f:
            lines = f.readlines()

        offset = max(1, args.get("offset", 1))
        limit = min(500, args.get("limit", 200))

        selected = lines[offset - 1 : offset - 1 + limit]
        total_lines = len(lines)

        numbered = []
        for i, line in enumerate(selected, start=offset):
            numbered.append(f"{i:4d} | {line.rstrip()}")

        result = "\n".join(numbered)
        if offset + limit < total_lines:
            result += f"\n\n... ({total_lines - offset - limit + 1} more lines, {total_lines} total)"

        return result

    def _tool_list_directory(self, temp_dir: str, args: dict) -> str:
        rel_path = args.get("path", ".")
        path = self._safe_path(temp_dir, rel_path)
        if not os.path.exists(path):
            return f"Directory not found: {rel_path}"
        if not os.path.isdir(path):
            return f"Not a directory: {rel_path}"

        entries = sorted(os.listdir(path))
        result_lines = []
        for entry in entries:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                result_lines.append(f"{entry}/")
            else:
                size = os.path.getsize(full)
                result_lines.append(f"{entry} ({size} bytes)")

        return "\n".join(result_lines) if result_lines else "(empty directory)"

    def _tool_search_files(self, temp_dir: str, args: dict) -> str:
        search_path = self._safe_path(temp_dir, args.get("path", "."))
        if not os.path.isdir(search_path):
            return f"Not a directory: {args.get('path', '.')}"

        pattern = args["pattern"]
        file_pattern = args.get("file_pattern", None)

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = []
        for root, _dirs, files in os.walk(search_path):
            # Skip hidden directories and common non-source dirs
            _dirs[:] = [d for d in _dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".git"}]

            for fname in files:
                if file_pattern:
                    import fnmatch

                    if not fnmatch.fnmatch(fname, file_pattern):
                        continue

                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, temp_dir)

                try:
                    with open(fpath, errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if compiled.search(line):
                                matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                                if len(matches) >= 50:
                                    return "\n".join(matches) + "\n\n(results capped at 50 matches)"
                except (OSError, UnicodeDecodeError):
                    continue

        if not matches:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(matches)

    def _tool_edit_file(self, temp_dir: str, args: dict) -> str:
        path = self._safe_path(temp_dir, args["path"])
        old_text = args["old_text"]
        new_text = args["new_text"]

        # Create new file
        if not old_text:
            if os.path.exists(path):
                return "Error: File already exists. Use old_text to specify what to replace, or read the file first."
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(new_text)

            # Lint check for Python files
            lint_error = self._check_syntax(path)
            if lint_error:
                os.remove(path)
                return f"Edit rejected — syntax error in new file:\n{lint_error}\nThe file was not created."

            return f"File created: {args['path']}"

        # Edit existing file
        if not os.path.exists(path):
            return f"File not found: {args['path']}"

        with open(path) as f:
            content = f.read()

        if old_text not in content:
            # Provide helpful context
            return (
                f"Error: old_text not found in {args['path']}. "
                "Make sure the text matches exactly, including whitespace and indentation. "
                "Read the file first to see the exact content."
            )

        # Check for ambiguous matches
        count = content.count(old_text)
        if count > 1:
            return (
                f"Error: old_text appears {count} times in {args['path']}. "
                "Provide more surrounding context to make the match unique."
            )

        # Apply the edit
        new_content = content.replace(old_text, new_text, 1)

        # Lint check for Python files before writing
        if path.endswith(".py"):
            # Write to temp location for syntax check
            temp_check = path + ".tmp"
            with open(temp_check, "w") as f:
                f.write(new_content)
            lint_error = self._check_syntax(temp_check)
            os.remove(temp_check)
            if lint_error:
                return f"Edit rejected — would introduce syntax error:\n{lint_error}\nThe file was not modified. Please fix the syntax and try again."

        with open(path, "w") as f:
            f.write(new_content)

        return f"Successfully edited {args['path']}"

    def _tool_run_command(self, temp_dir: str, args: dict) -> str:
        command = args["command"]

        # Basic safety: prevent obvious escapes
        if any(dangerous in command for dangerous in ["rm -rf /", "dd if=", "mkfs", "> /dev/"]):
            return "Command rejected for safety reasons."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "Command timed out after 120 seconds."

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr

        # Cap output
        if len(output) > 3000:
            output = output[:1500] + "\n\n... (output truncated) ...\n\n" + output[-1500:]

        return f"Exit code: {result.returncode}\n{output}" if output else f"Exit code: {result.returncode}"

    def _check_syntax(self, filepath: str) -> str | None:
        """Check Python syntax. Returns error message or None if valid."""
        try:
            with open(filepath) as f:
                source = f.read()
            compile(source, filepath, "exec")
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"

    def _safe_path(self, temp_dir: str, relative_path: str) -> str:
        """Prevent path traversal attacks."""
        resolved = os.path.realpath(os.path.join(temp_dir, relative_path))
        real_temp = os.path.realpath(temp_dir)
        if not resolved.startswith(real_temp + os.sep) and resolved != real_temp:
            raise ValueError(f"Path traversal detected: {relative_path}")
        return resolved

    def _detect_new_files(self, temp_dir: str, original_contents: dict[str, str]) -> dict[str, str]:
        """Detect files created by the agent that weren't in the original file set."""
        new_files: dict[str, str] = {}
        original_paths = set(original_contents.keys())

        # Walk the temp dir looking for source files not in the original set
        source_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".rb",
            ".c",
            ".cpp",
            ".h",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".cfg",
            ".ini",
            ".txt",
            ".md",
        }
        for root, dirs, files in os.walk(temp_dir):
            # Skip .git and other hidden/generated dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".tox"}]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, temp_dir)
                _, ext = os.path.splitext(fname)
                if rel not in original_paths and ext in source_extensions:
                    try:
                        with open(fpath) as f:
                            new_files[rel] = f.read()
                    except (OSError, UnicodeDecodeError):
                        continue

        return new_files

    def _extract_summary(self, messages: list[dict]) -> str:
        """Extract the agent's final text summary from the conversation."""
        # Look at the last assistant message for text content
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, str):
                    return content[:500]
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if hasattr(block, "type") and block.type == "text":
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block["text"])
                    if text_parts:
                        return "\n".join(text_parts)[:500]
        return "Agent made changes but did not provide a summary."
