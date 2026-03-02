"""
Agentic mutator for the spec_task problem type.

Similar to the repo_task agent but tailored for greenfield development:
the prompt focuses on implementing a specification rather than fixing tests.
"""

from __future__ import annotations

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problems.repo_task_agent import AgenticRepoMutator
from darwinian_evolver.problems.repo_task_agent import build_repo_structure_map
from darwinian_evolver.problems.spec_task import SpecTaskFailureCase
from darwinian_evolver.problems.spec_task import SpecTaskOrganism

SPEC_AGENT_SYSTEM_PROMPT = """You are an expert software engineer implementing code to meet a specification.

Your goal is to write or improve code so it fully satisfies the given specification.

## Approach

1. **Read the specification carefully** — understand exactly what's required.
2. **Explore existing code** — read what's already there before making changes.
3. **Implement incrementally** — make focused changes, one at a time.
4. **Validate your work** — if a validation command is mentioned, run it.
5. **Address all criteria** — pay attention to any specific feedback from previous reviews.

## Rules

- Write clean, well-structured code.
- Handle edge cases and errors appropriately.
- Do NOT add unnecessary complexity or over-engineer.
- Do NOT leave placeholder code — implement fully.

## When you're done

Provide a brief summary of what you implemented and how it addresses the specification.
"""


class AgenticSpecMutator(AgenticRepoMutator):
    """Mutator for spec_task that generates/improves code based on a specification."""

    def _build_prompt(
        self,
        organism: SpecTaskOrganism,
        failure_cases: list[SpecTaskFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> str:
        parts = []

        # Specification
        parts.append(f"## Specification\n\n{organism.spec}")

        # Task description (additional context)
        if organism.task_description:
            parts.append(f"## Task context\n\n{organism.task_description}")

        # Current code structure
        if organism.file_contents:
            structure_map = build_repo_structure_map(organism.file_contents)
            parts.append(f"## Current code structure\n\n```\n{structure_map}\n```")

        # Judge feedback (failure cases)
        if failure_cases:
            parts.append("## Review feedback\n\nThe following criteria need improvement:")
            for fc in failure_cases:
                parts.append(f"### {fc.criterion}")
                if fc.feedback:
                    parts.append(f"{fc.feedback}")

        # Learning log
        if learning_log_entries:
            parts.append("## Previous attempts\n\nLearn from these previous attempts:")
            for entry in learning_log_entries[-5:]:
                parts.append(f"- **Attempted:** {entry.attempted_change}")
                parts.append(f"  **Outcome:** {entry.observed_outcome}")

        # Instructions
        parts.append(
            "## Instructions\n\n"
            "1. Read the existing code to understand the current state.\n"
            "2. Implement the changes needed to address the review feedback.\n"
            "3. Make sure your implementation is complete — no TODOs or placeholders.\n"
            "4. When done, provide a brief summary of your changes."
        )

        return "\n\n".join(parts)

    def mutate(
        self,
        organism: SpecTaskOrganism,
        failure_cases: list[SpecTaskFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[SpecTaskOrganism]:
        from darwinian_evolver.git_based_problem import GitBasedOrganism

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
                        system=SPEC_AGENT_SYSTEM_PROMPT,
                        tools=self._get_tools(),
                        messages=messages,
                    )
                except Exception as e:
                    print(f"[SpecMutator] API error: {e}")
                    break

                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    mutation_input_tokens += response.usage.input_tokens
                    mutation_output_tokens += response.usage.output_tokens

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    break

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

            new_file_contents = GitBasedOrganism.capture_modified_files(temp_dir, organism.file_contents)
            new_files = self._detect_new_files(temp_dir, organism.file_contents)
            new_file_contents.update(new_files)

            summary = self._extract_summary(messages)

        # Update cumulative cost tracking
        self.total_input_tokens += mutation_input_tokens
        self.total_output_tokens += mutation_output_tokens
        self.total_api_calls += 1

        if new_file_contents == organism.file_contents and not new_files:
            return []

        return [
            SpecTaskOrganism(
                repo_root=organism.repo_root,
                git_hash=organism.git_hash,
                file_contents=new_file_contents,
                task_description=organism.task_description,
                spec=organism.spec,
                from_change_summary=summary,
            )
        ]

    @staticmethod
    def _get_tools():
        from darwinian_evolver.problems.repo_task_agent import TOOLS

        return TOOLS
