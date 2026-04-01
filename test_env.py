import requests
import json
import threading
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from groq import Groq

BASE_URL = "https://bhoomiladia-coder-env.hf.space"
client = Groq(api_key="groq_api")

SYSTEM_PROMPT = """You are a code fixing agent. Your ONLY job is to fix technical debt in Python files.

STRICT RULES:
- Do NOT add new features, loops, input() calls, or UI improvements
- Do NOT change code structure beyond what is necessary to fix the bug
- Fix ONLY the specific security, logic, or syntax issue present
- Make minimal changes to match the intended fix

AVAILABLE COMMANDS:
- LIST_FILES: See all files in the workspace
- READ_FILE: Read a specific file's contents
- RUN_LINTER: Check a file for syntax errors
- RUN_TESTS: Run all checks (syntax + security + logic) on workspace
- WRITE_FILE: Overwrite a file with your fixed version

Respond ONLY with raw JSON (no markdown, no code fences):
{
    "command": "LIST_FILES | READ_FILE | RUN_LINTER | RUN_TESTS | WRITE_FILE",
    "target_file": "filename.py or empty string for LIST_FILES/RUN_TESTS",
    "patch_content": "fixed code (only for WRITE_FILE, else empty string)",
    "explanation": "exactly what debt you are fixing and why"
}"""


def call_groq(conversation: list) -> dict | None:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=900,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation
            )
            raw = response.choices[0].message.content
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            action = json.loads(cleaned)
            return action, raw
        except json.JSONDecodeError:
            print(f"  [Retry {attempt + 1}/3] JSON parse failed, retrying...")
    return None, None


def format_diagnostics(diagnostics: list) -> str:
    if not diagnostics:
        return "  No issues detected."
    lines = []
    for d in diagnostics:
        lines.append(f"  [{d['severity'].upper()}] {d['file']}:{d.get('line','?')} — {d['message']}")
    return "\n".join(lines)


def format_reward_breakdown(breakdown: dict) -> str:
    if not breakdown:
        return ""
    parts = [f"{k}: {v:.2f}" for k, v in breakdown.items()]
    return "  Breakdown → " + " | ".join(parts)


def run_agent(task_id: int = 0, max_steps: int = 15):
    print(f"\n{'='*60}")
    print(f"  Task {task_id} — CodeDebt RL Agent")
    print(f"  Server: {BASE_URL}")
    print(f"{'='*60}\n")

    # ── 1. Reset ──────────────────────────────────────────────────
    try:
        res = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not reach server: {e}")
        return

    obs = res.json()
    print("── Initial Observation ──────────────────────────────────")
    print(f"  {obs['observation']['terminal_output']}")
    print(f"\n  Files:")
    for fname, code in obs['observation'].get('code_view', {}).items():
        print(f"\n  [{fname}]\n{code}")
    print(f"\n  Diagnostics:")
    print(format_diagnostics(obs['observation'].get('diagnostics', [])))
    print()

    conversation = []
    prev_reward = 0.0

    # ── 2. Agent loop ─────────────────────────────────────────────
    for step in range(1, max_steps + 1):
        print(f"── Step {step}/{max_steps} {'─'*40}")

        reward_delta = obs['reward'] - prev_reward
        direction = "⬆ improving" if reward_delta > 0 else ("⬇ worse" if reward_delta < 0 else "→ no change")

        current_message = f"""Current workspace files:
{json.dumps(obs['observation'].get('code_view', {}), indent=2)}

Last terminal output: {obs['observation']['terminal_output']}
Current reward: {obs['reward']:.4f} ({direction} from {prev_reward:.4f})
Task goal: {obs['observation']['goal']}

Active diagnostics:
{format_diagnostics(obs['observation'].get('diagnostics', []))}

What is your next action? Remember: fix debt only, no new features. Respond in raw JSON."""

        conversation.append({"role": "user", "content": current_message})

        # ── call LLM ──
        action, raw = call_groq(conversation)
        if action is None:
            print("  [SKIP] Could not parse a valid JSON action after 3 attempts.")
            conversation.append({"role": "assistant", "content": "{}"})
            continue

        conversation.append({"role": "assistant", "content": raw})

        print(f"  Command   : {action.get('command')}")
        print(f"  Target    : {action.get('target_file') or '—'}")
        print(f"  Reason    : {action.get('explanation', '')[:120]}")

        # ── call env ──
        try:
            res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=30
            )
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Step request failed: {e}")
            break

        prev_reward = obs['reward']
        obs = res.json()

        print(f"  Terminal  : {obs['observation']['terminal_output']}")
        print(f"  Reward    : {obs['reward']:.4f}")
        print(format_reward_breakdown(obs['observation'].get('reward_breakdown', {})))

        if obs['observation'].get('diagnostics'):
            print(f"  Diagnostics:")
            print(format_diagnostics(obs['observation']['diagnostics']))

        # ── termination ──
        if obs['done']:
            if obs['reward'] >= 1.0:
                print(f"\n✅ Task {task_id} solved in {step} steps! Final reward: {obs['reward']:.4f}")
            else:
                print(f"\n⏱ Max steps reached. Final reward: {obs['reward']:.4f}")
            return

        print()

    print(f"\n⏱ Agent loop ended. Final reward: {obs['reward']:.4f}")


def run_all_tasks(task_ids: list[int] = None):
    if task_ids is None:
        task_ids = list(range(6))
    results = []
    for tid in task_ids:
        print(f"\n{'#'*60}")
        run_agent(task_id=tid)
        results.append(tid)
    print(f"\n{'='*60}")
    print(f"  Finished {len(results)} tasks.")
    print(f"{'='*60}")


# ── GUI Implementation ────────────────────────────────────────────────────────

class ConsoleRedirector:
    """Redirects print statements to the Tkinter Text widget safely."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)

    def flush(self):
        pass

class AgentTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CodeDebt Agent Tester")
        self.root.geometry("850x650")
        
        # Style configurations
        self.font_main = ("Consolas", 10)
        self.root.configure(padx=10, pady=10)

        # 1. Top Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(control_frame, text="Select Task:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.task_var = tk.StringVar(value="Task 0")
        self.task_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.task_var, 
            state="readonly", 
            width=15,
            values=[f"Task {i}" for i in range(6)] + ["Run All (0-5)"]
        )
        self.task_dropdown.pack(side=tk.LEFT)

        self.run_btn = tk.Button(control_frame, text="▶ Run Agent", bg="#4CAF50", fg="black", command=self.start_thread)
        self.run_btn.pack(side=tk.LEFT, padx=15)

        self.clear_btn = tk.Button(control_frame, text="Clear Console", command=self.clear_console)
        self.clear_btn.pack(side=tk.LEFT)

        # Status Label
        self.status_label = tk.Label(control_frame, text="Status: Idle", fg="gray")
        self.status_label.pack(side=tk.RIGHT)

        # 2. Console Output Area
        self.console = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=self.font_main, bg="#1E1E1E", fg="#D4D4D4")
        self.console.pack(expand=True, fill=tk.BOTH)
        self.console.configure(state=tk.DISABLED)

        # Redirect standard output
        sys.stdout = ConsoleRedirector(self.console)
        sys.stderr = ConsoleRedirector(self.console)

    def clear_console(self):
        self.console.configure(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.configure(state=tk.DISABLED)

    def start_thread(self):
        """Disables the run button and starts the agent logic in a background thread."""
        self.run_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Running...", fg="blue")
        
        selection = self.task_var.get()
        
        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=self.execute_agent, args=(selection,), daemon=True)
        thread.start()

    def execute_agent(self, selection):
        try:
            if selection == "Run All (0-5)":
                run_all_tasks([0, 1, 2, 3, 4, 5])
            else:
                task_id = int(selection.replace("Task ", ""))
                run_agent(task_id=task_id)
        except Exception as e:
            print(f"\n[SYSTEM ERROR] {e}")
        finally:
            # Re-enable UI safely
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.run_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Idle", fg="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentTesterGUI(root)
    root.mainloop()