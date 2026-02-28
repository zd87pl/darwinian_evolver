"""This file contains an extended Organism base class that can be used for problems that operate on a git repository."""

from __future__ import annotations

import contextlib
import difflib
import os
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Generator
from typing import Iterable
from typing import Self

from pydantic import Field
from pydantic import computed_field

from darwinian_evolver.problem import Organism


class GitBasedOrganism(Organism):
    """
    Organism base class that represents a modified git repository.

    The state of the organism is represented by a git commit hash and a set of file contents that might have been modified on top of the commit.
    """

    repo_root: str
    git_hash: str
    file_contents: dict[str, str]
    deleted_files: set[str] = Field(default_factory=set)

    @classmethod
    def make_initial_organism_from_repo(
        cls,
        repo_root: str,
        files_to_capture: Iterable[str],
        git_hash: str | None = None,
        **kwargs,
    ) -> Self:
        """
        Initialize the organism with the current state of the git repository.

        This method captures the current git commit hash and the contents of specified files.
        """
        # Change working directory to the repository root
        with contextlib.chdir(repo_root):
            if git_hash is None:
                # Populate the git hash from the current repository
                git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

            file_contents = {file_path: cls._get_file_content(git_hash, file_path) for file_path in files_to_capture}

        return cls(repo_root=repo_root, git_hash=git_hash, file_contents=file_contents, **kwargs)

    @computed_field
    @property
    def diff_from_parent(self) -> str:
        """
        Generate a diff between this organism and its parent.

        The diff is primarily intended for visual inspection to better understand what changes occurred.
        """
        if self.parent is None:
            return ""

        assert isinstance(self.parent, GitBasedOrganism), "Parent must be a GitBasedOrganism as well"

        diffs_per_file = []

        # Diff files present in both parent and child
        all_files = set(self.file_contents.keys()) | set(self.parent.file_contents.keys())
        for file_path in sorted(all_files):
            parent_content = self.parent.file_contents.get(file_path, "")
            child_content = self.file_contents.get(file_path, "")

            if file_path in self.deleted_files:
                child_content = ""

            if parent_content != child_content:
                file_diff = difflib.unified_diff(
                    parent_content.splitlines(keepends=True),
                    child_content.splitlines(keepends=True),
                    fromfile=file_path,
                    tofile=file_path,
                )
                file_diff_str = "\n".join(file_diff)
                diffs_per_file.append(file_diff_str)

        return "\n".join(diffs_per_file)

    @contextmanager
    def build_repo(self) -> Generator[str, None, None]:
        """
        Build a temporary git repository with the contents of this organism.

        Intended to be used as a context manager: `with organism.build_repo() as temp_dir: ...`

        This is useful for running evaluations or mutations within the context of this organism.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", self.repo_root, temp_dir], check=True, capture_output=True)
            subprocess.run(["git", "checkout", self.git_hash], cwd=temp_dir, check=True, capture_output=True)

            # Replace the relevant files with the contents from this organism
            for file_path, content in self.file_contents.items():
                temp_file_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                with open(temp_file_path, "w") as f:
                    f.write(content)

            # Remove deleted files
            for file_path in self.deleted_files:
                temp_file_path = os.path.join(temp_dir, file_path)
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            yield temp_dir

    @staticmethod
    def capture_modified_files(temp_dir: str, base_file_contents: dict[str, str]) -> dict[str, str]:
        """
        Capture files that have been modified in the temp directory relative to the base file contents.

        Returns a new file_contents dict with updated content for any changed files,
        plus any new files created by the agent.
        """
        new_file_contents = {}
        for file_path, original_content in base_file_contents.items():
            full_path = os.path.join(temp_dir, file_path)
            if os.path.exists(full_path):
                with open(full_path) as f:
                    new_file_contents[file_path] = f.read()
            # If file was deleted, don't include it (handled via deleted_files)

        return new_file_contents

    @staticmethod
    def _get_file_content(git_hash: str, file_path: str) -> str:
        """Get the contents of a file in the current git repository."""
        return subprocess.check_output(["git", "show", f"{git_hash}:{file_path}"]).decode("utf-8")
