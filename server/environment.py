"""CodeDebtRefactor RL Environment — core environment logic.

Implements the OpenEnv ``Environment`` interface for training agents to
detect and remediate technical debt in Python source files.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import difflib

from openenv.core.env_server import Environment

from models import FixCodeAction, TerminalObservation, CurrentState

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TASKS_PATH = Path(__file__).resolve().parent.parent / "tasks.json"

# Persistent state file — survives across requests on HuggingFace Spaces
_STATE_PATH = Path("/tmp/env_state.json")

# ---------------------------------------------------------------------------
# Reward weights (must sum to 1.0)
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "syntax": 0.15,
    "security": 0.25,
    "logic": 0.20,
    "similarity": 0.40,
}


class CodeDebtEnvironment(Environment):
    """An RL environment that presents broken Python files and rewards the
    agent for fixing syntax errors, security vulnerabilities, and logic bugs.

    State is persisted to disk so that it survives across requests even when
    the server creates a new instance per request (e.g. HuggingFace Spaces).
    """

    def __init__(self) -> None:
        super().__init__()
        self._tasks: List[Dict[str, Any]] = self._load_tasks()
        self._state = CurrentState()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        """Persist current state to disk."""
        data = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "files": self._state.files,
            "hidden_solution": self._state.hidden_solution,
            "max_steps": self._state.max_steps,
            "task_id": self._state.task_id,
            "task_description": self._state.task_description,
        }
        try:
            with open(_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.debug("State saved to %s", _STATE_PATH)
        except OSError as exc:
            logger.error("Failed to save state: %s", exc)

    def _load_state(self) -> bool:
        """Load persisted state from disk. Returns True if successful."""
        if not _STATE_PATH.exists():
            logger.warning("No persisted state found at %s", _STATE_PATH)
            return False
        try:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._state = CurrentState(
                episode_id=data.get("episode_id"),
                step_count=data.get("step_count", 0),
                files=data.get("files", {}),
                hidden_solution=data.get("hidden_solution", {}),
                max_steps=data.get("max_steps", 15),
                task_id=data.get("task_id", 0),
                task_description=data.get("task_description", ""),
            )
            logger.debug("State loaded from %s", _STATE_PATH)
            return True
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load state: %s", exc)
            return False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------
    @property
    def state(self) -> CurrentState:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TerminalObservation:
        """Reset the environment to a fresh task.

        Accepts ``task_id`` as a keyword arg (default 0).
        """
        task_id: int = kwargs.get("task_id", 0)

        # --- validate ---
        if not 0 <= task_id < len(self._tasks):
            return TerminalObservation(
                terminal_output=(
                    f"Error: task_id {task_id} out of range "
                    f"[0, {len(self._tasks) - 1}]."
                ),
                done=True,
                reward=0.0,
                last_action_error=f"Invalid task_id: {task_id}",
                available_files=[],
            )

        task = self._tasks[task_id]

        # --- reset state ---
        self._state = CurrentState(
            episode_id=episode_id,
            step_count=0,
            files=dict(task["initial_code"]),
            hidden_solution=dict(task["solution"]),
            max_steps=kwargs.get("max_steps", 15),
            task_id=task_id,
            task_description=task.get("description", ""),
        )

        # Persist immediately so step() on a fresh instance can load it
        self._save_state()

        initial_diags = self._run_all_checks(self._state.files)

        logger.info(
            "Environment reset — task %d (%s)", task_id, task.get("difficulty")
        )

        return TerminalObservation(
            terminal_output=(
                f"Environment reset. Task {task_id} loaded "
                f"({task.get('difficulty', '?')} difficulty).\n"
                f"Goal: {task.get('description', 'Fix the code.')}"
            ),
            code_view=dict(self._state.files),
            reward=0.0,
            done=False,
            goal=task.get("description", "Fix the technical debt."),
            last_action_error=None,
            diagnostics=initial_diags,
            available_files=list(self._state.files.keys()),
        )

    def step(
        self,
        action: FixCodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TerminalObservation:
        """Execute one agent action and return the new observation."""

        # --- reload persisted state (handles fresh instance per request) ---
        if not self._state.files and not self._load_state():
            return TerminalObservation(
                terminal_output=(
                    "Error: no active session. Please call /reset first."
                ),
                done=True,
                reward=0.0,
                last_action_error="Session not initialised.",
                available_files=[],
            )

        self._state.step_count += 1
        terminal_output = ""
        last_error: Optional[str] = None

        logger.info(
            "Step %d — command=%s target=%s",
            self._state.step_count,
            action.command,
            action.target_file,
        )

        # ---- dispatch command ----
        if action.command == "WRITE_FILE":
            terminal_output, last_error = self._handle_write(action)
        elif action.command == "READ_FILE":
            terminal_output, last_error = self._handle_read(action)
        elif action.command == "RUN_LINTER":
            terminal_output, last_error = self._handle_linter(action)
        elif action.command == "RUN_TESTS":
            terminal_output, last_error = self._handle_tests(action)
        elif action.command == "LIST_FILES":
            terminal_output, last_error = self._handle_list()
        else:
            terminal_output = f"Unknown command: {action.command}"
            last_error = terminal_output

        # ---- compute reward ----
        reward, breakdown = self._compute_reward(
            self._state.files, self._state.hidden_solution
        )

        # ---- diagnostics ----
        diagnostics = self._run_all_checks(self._state.files)

        # ---- termination ----
        done = (reward >= 1.0) or (
            self._state.step_count >= self._state.max_steps
        )

        if done and reward < 1.0:
            terminal_output += "\n⏱ Step budget exhausted."
        elif done:
            terminal_output += "\n✅ All issues resolved! Task complete."

        # --- persist updated state ---
        self._save_state()

        return TerminalObservation(
            terminal_output=terminal_output,
            code_view=dict(self._state.files),
            reward=reward,
            done=done,
            goal=self._state.task_description,
            last_action_error=last_error,
            diagnostics=diagnostics,
            reward_breakdown=breakdown,
            available_files=list(self._state.files.keys()),
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _handle_write(
        self, action: FixCodeAction
    ) -> tuple[str, Optional[str]]:
        if not action.target_file:
            return ("", "WRITE_FILE requires a target_file.")
        if action.target_file not in self._state.files:
            return (
                "",
                f"File '{action.target_file}' does not exist. "
                f"Available: {list(self._state.files.keys())}",
            )
        self._state.files[action.target_file] = action.patch_content
        return (f"Successfully wrote to {action.target_file}.", None)

    def _handle_read(
        self, action: FixCodeAction
    ) -> tuple[str, Optional[str]]:
        if not action.target_file:
            return ("", "READ_FILE requires a target_file.")
        code = self._state.files.get(action.target_file)
        if code is None:
            return (
                "",
                f"File '{action.target_file}' not found. "
                f"Available: {list(self._state.files.keys())}",
            )
        return (f"--- {action.target_file} ---\n{code}", None)

    def _handle_linter(
        self, action: FixCodeAction
    ) -> tuple[str, Optional[str]]:
        if not action.target_file:
            return ("", "RUN_LINTER requires a target_file.")
        code = self._state.files.get(action.target_file)
        if code is None:
            return ("", f"File '{action.target_file}' not found.")
        output = self._run_virtual_linter(action.target_file, code)
        return (output, None)

    def _handle_tests(
        self, action: FixCodeAction
    ) -> tuple[str, Optional[str]]:
        """Run virtual tests for a specific file (or all files)."""
        task = self._tasks[self._state.task_id]
        test_cases = task.get("tests", {})

        if not test_cases:
            # Fallback: run linter + checks on all files
            parts: list[str] = []
            for fname, code in self._state.files.items():
                parts.append(self._run_virtual_linter(fname, code))
                parts.extend(
                    f"  [{d['severity']}] {d['message']}"
                    for d in self._check_security(fname, code)
                    + self._check_logic(fname, code)
                )
            return ("\n".join(parts) or "All checks passed.", None)

        # Explicit test cases
        results: list[str] = []
        passed = 0
        total = 0
        target = action.target_file or None
        for fname, cases in test_cases.items():
            if target and fname != target:
                continue
            code = self._state.files.get(fname, "")
            for tc in cases:
                total += 1
                if self._virtual_test(code, tc):
                    passed += 1
                    results.append(f"  ✅ {tc['name']}")
                else:
                    results.append(
                        f"  ❌ {tc['name']}: {tc.get('hint', 'failed')}"
                    )

        header = f"Tests: {passed}/{total} passed"
        return (header + "\n" + "\n".join(results), None)

    def _handle_list(self) -> tuple[str, Optional[str]]:
        names = list(self._state.files.keys())
        if not names:
            return (
                "No files in workspace. Did you call /reset first?",
                "Workspace is empty — state may not have been loaded correctly.",
            )
        return (
            "Files in workspace:\n" + "\n".join(f"  - {n}" for n in names),
            None,
        )

    # ------------------------------------------------------------------
    # Virtual linter (syntax only)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_virtual_linter(filename: str, code: str) -> str:
        if not code.strip():
            return f"{filename}: WARNING — file is empty."
        try:
            ast.parse(code, filename=filename)
            return f"{filename}: ✅ syntax OK"
        except SyntaxError as exc:
            return f"{filename}:{exc.lineno}: SyntaxError: {exc.msg}"

    # ------------------------------------------------------------------
    # Security checker
    # ------------------------------------------------------------------
    @staticmethod
    def _check_security(filename: str, code: str) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []

        patterns = [
            (
                r"""(['"]SELECT\s.*?['"])\s*\+""",
                "SQL injection risk: string-concatenated query detected. "
                "Use parameterized queries.",
                "high",
            ),
            (
                r"hashlib\.md5\b",
                "Insecure hash: MD5 is broken for security use. "
                "Use SHA-256 or bcrypt.",
                "high",
            ),
            (
                r"""(?i)(secret[_\s]*key|password|api[_\s]*key)\s*=\s*['"][^'"]{4,}['"]""",
                "Hardcoded secret detected. Use environment variables.",
                "high",
            ),
            (
                r"\bDEBUG\s*=\s*True\b",
                "DEBUG=True should not be hardcoded. Use env vars.",
                "medium",
            ),
            (
                r"""ALLOWED_HOSTS\s*=\s*['"]?\*['"]?""",
                "ALLOWED_HOSTS='*' allows all origins. Restrict in production.",
                "medium",
            ),
            (
                r"\brandom\.choices?\b",
                "Using 'random' for token generation is not cryptographically "
                "secure. Use the 'secrets' module.",
                "high",
            ),
        ]

        for pattern, message, severity in patterns:
            for match in re.finditer(pattern, code):
                lineno = code[: match.start()].count("\n") + 1
                issues.append(
                    {
                        "file": filename,
                        "line": str(lineno),
                        "severity": severity,
                        "category": "security",
                        "message": message,
                    }
                )

        return issues

    # ------------------------------------------------------------------
    # Logic checker
    # ------------------------------------------------------------------
    @staticmethod
    def _check_logic(filename: str, code: str) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []

        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None and isinstance(
                        default, (ast.List, ast.Dict, ast.Set)
                    ):
                        issues.append(
                            {
                                "file": filename,
                                "line": str(node.lineno),
                                "severity": "medium",
                                "category": "logic",
                                "message": (
                                    f"Function '{node.name}' has a mutable default "
                                    f"argument. Use None and initialize inside the body."
                                ),
                            }
                        )

                if not _has_return(node) and _has_assignments(node):
                    issues.append(
                        {
                            "file": filename,
                            "line": str(node.lineno),
                            "severity": "medium",
                            "category": "logic",
                            "message": (
                                f"Function '{node.name}' performs computation "
                                f"but never returns a value."
                            ),
                        }
                    )

            if isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Attribute
            ):
                if node.value.attr == "close":
                    issues.append(
                        {
                            "file": filename,
                            "line": str(node.lineno),
                            "severity": "high",
                            "category": "logic",
                            "message": (
                                "'.close' referenced but not called. "
                                "Did you mean '.close()'?"
                            ),
                        }
                    )

        return issues

    # ------------------------------------------------------------------
    # Run ALL checks
    # ------------------------------------------------------------------
    def _run_all_checks(self, files: Dict[str, str]) -> List[Dict[str, str]]:
        all_diags: List[Dict[str, str]] = []
        for fname, code in files.items():
            try:
                ast.parse(code, filename=fname)
            except SyntaxError as exc:
                all_diags.append(
                    {
                        "file": fname,
                        "line": str(exc.lineno or 0),
                        "severity": "error",
                        "category": "syntax",
                        "message": f"SyntaxError: {exc.msg}",
                    }
                )
            all_diags.extend(self._check_security(fname, code))
            all_diags.extend(self._check_logic(fname, code))
        return all_diags

    # ------------------------------------------------------------------
    # Multi-signal reward
    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        current_files: Dict[str, str],
        goal_files: Dict[str, str],
    ) -> tuple[float, Dict[str, float]]:
        if not goal_files:
            return (0.0, {})

        n = len(goal_files)
        syntax_ok = 0
        security_ok = 0
        logic_ok = 0
        total_sim = 0.0

        for fname, goal_code in goal_files.items():
            code = current_files.get(fname, "")

            try:
                ast.parse(code, filename=fname)
                syntax_ok += 1
            except SyntaxError:
                pass

            if not self._check_security(fname, code):
                security_ok += 1

            if not self._check_logic(fname, code):
                logic_ok += 1

            seq = difflib.SequenceMatcher(None, code.strip(), goal_code.strip())
            total_sim += seq.ratio()

        breakdown = {
            "syntax": syntax_ok / n,
            "security": security_ok / n,
            "logic": logic_ok / n,
            "similarity": total_sim / n,
        }

        total = sum(REWARD_WEIGHTS[k] * breakdown[k] for k in REWARD_WEIGHTS)
        total = max(0.0, min(1.0, total))

        return (round(total, 4), {k: round(v, 4) for k, v in breakdown.items()})

    # ------------------------------------------------------------------
    # Virtual test runner
    # ------------------------------------------------------------------
    @staticmethod
    def _virtual_test(code: str, test_case: Dict[str, Any]) -> bool:
        if test_case.get("parses", False):
            try:
                ast.parse(code)
            except SyntaxError:
                return False

        for pattern in test_case.get("contains", []):
            if pattern not in code:
                return False

        for pattern in test_case.get("not_contains", []):
            if pattern in code:
                return False

        return True

    # ------------------------------------------------------------------
    # Task loader
    # ------------------------------------------------------------------
    @staticmethod
    def _load_tasks() -> List[Dict[str, Any]]:
        try:
            with open(_TASKS_PATH, "r", encoding="utf-8") as fh:
                tasks = json.load(fh)
            logger.info("Loaded %d tasks from %s", len(tasks), _TASKS_PATH)
            return tasks
        except FileNotFoundError:
            logger.error("tasks.json not found at %s", _TASKS_PATH)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in tasks.json: %s", exc)
            return []


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _has_return(func_node: ast.FunctionDef) -> bool:
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value is not None:
            return True
    return False


def _has_assignments(func_node: ast.FunctionDef) -> bool:
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    return True
    return False


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
codeDebtEnvironment = CodeDebtEnvironment