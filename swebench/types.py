"""
Type definitions for swebench.
"""

from typing import TypedDict
from dataclasses import dataclass


class SWEbenchInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for evaluation of a single instance of SWE-bench.
    Assumes images are already built and available.
    """

    instance_id: str
    image: str
    eval_script_list: list[str]
    repo: str
    version: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]

    @property
    def eval_script(self):
        # TODO: Clean up this debug code - either remove if issue is fixed, or replace prints with proper logging
        # Debug: Check for nested lists in eval_script_list
        for i, item in enumerate(self.eval_script_list):
            if not isinstance(item, str):
                print(
                    f"ERROR: Item {i} in eval_script_list is not a string: {type(item)} = {repr(item)}"
                )
                print(f"Full eval_script_list: {repr(self.eval_script_list)}")
                if isinstance(item, list):
                    # Flatten the nested list
                    print(f"Flattening nested list at position {i}")
                    flattened = []
                    for j, x in enumerate(self.eval_script_list):
                        if isinstance(x, list):
                            flattened.extend(x)
                        else:
                            flattened.append(x)
                    self.eval_script_list = flattened
                    break

        return (
            "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list)
            + "\n"
        )
