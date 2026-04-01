"""Data models for the CodeDebtRefactor RL environment.

Defines the Action, Observation, and State types used by the environment
following the OpenEnv framework's Pydantic-based model system.
"""

from typing import Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# State — internal environment bookkeeping
# ---------------------------------------------------------------------------
class CurrentState(State):
    """Tracks the mutable state of the current episode.

    Attributes:
        files: Mapping of virtual filenames → current source code.
        hidden_solution: The ground-truth solution the agent is working toward.
        max_steps: Hard step budget per episode.
        task_id: The active task identifier.
        task_description: Human-readable description of the current task.
    """

    files: Dict[str, str] = Field(default_factory=dict)
    hidden_solution: Dict[str, str] = Field(default_factory=dict)
    max_steps: int = Field(default=15, ge=1)
    task_id: int = Field(default=0, ge=0)
    task_description: str = Field(default="")


# ---------------------------------------------------------------------------
# Observation — what the agent sees after each action
# ---------------------------------------------------------------------------
class TerminalObservation(Observation):
    """Observation returned to the agent after every step (and on reset).

    Inherits ``done: bool`` and ``reward`` from the OpenEnv base ``Observation``.

    Attributes:
        terminal_output: Simulated stdout/stderr from the last action.
        code_view: Current snapshot of all virtual files.
        goal: Natural-language description of the task.
        last_action_error: If the last action was invalid, explains why.
        diagnostics: Structured list of issues detected by the checker suite.
        reward_breakdown: Per-signal reward decomposition for interpretability.
        available_files: List of filenames the agent can interact with.
    """

    terminal_output: str = Field(default="")
    code_view: Dict[str, str] = Field(default_factory=dict)
    goal: str = Field(default="Fix the technical debt in the provided files.")
    last_action_error: Optional[str] = Field(default=None)
    diagnostics: List[Dict[str, str]] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    available_files: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Action — what the agent can do
# ---------------------------------------------------------------------------
class FixCodeAction(Action):
    """An action the agent submits to the environment.

    Supported commands:
        WRITE_FILE  — overwrite ``target_file`` with ``patch_content``.
        READ_FILE   — request the contents of ``target_file``.
        RUN_LINTER  — run the syntax + security + logic checker on ``target_file``.
        RUN_TESTS   — run the virtual test suite for ``target_file``.
        LIST_FILES  — list all virtual files in the current episode.

    Attributes:
        command: One of the supported action verbs.
        target_file: Which virtual file the action targets (ignored for LIST_FILES).
        patch_content: New file contents (only used by WRITE_FILE).
        explanation: Free-text reasoning from the agent (logged, not graded).
    """

    command: Literal[
        "WRITE_FILE", "READ_FILE", "RUN_LINTER", "RUN_TESTS", "LIST_FILES"
    ] = Field(default="WRITE_FILE")
    target_file: str = Field(default="")
    patch_content: str = Field(default="")
    explanation: str = Field(default="")


# ---------------------------------------------------------------------------
# Backward-compatible aliases (to avoid breaking existing client code)
# ---------------------------------------------------------------------------
currentState = CurrentState
terminalObservation = TerminalObservation
fixCodeAction = FixCodeAction