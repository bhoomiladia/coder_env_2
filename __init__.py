"""CodeDebtRefactor — an OpenEnv RL environment for technical debt remediation."""

from .server.environment import CodeDebtEnvironment
from .models import FixCodeAction, TerminalObservation, CurrentState

# Backward-compatible aliases
from .server.environment import codeDebtEnvironment
from .models import fixCodeAction, terminalObservation, currentState

__all__ = [
    # Canonical (PascalCase)
    "CodeDebtEnvironment",
    "FixCodeAction",
    "TerminalObservation",
    "CurrentState",
    # Legacy aliases
    "codeDebtEnvironment",
    "fixCodeAction",
    "terminalObservation",
    "currentState",
]