"""Quick test"""
import subprocess
import sys
import os

# Override ENV_URL to a non-existent server so we fail fast
env = os.environ.copy()
env["ENV_URL"] = "http://127.0.0.1:1"  # Instant connection refused

result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True,
    text=True,
    timeout=60,
    cwd=os.path.dirname(os.path.abspath(__file__)),
    env=env,
)

print("=== STDOUT ===")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")

# Check for markers
has_start = "[START]" in result.stdout
has_end = "[END]" in result.stdout
print(f"\n[START] found: {has_start}")
print(f"[END] found: {has_end}")

start_count = result.stdout.count("[START]")
end_count = result.stdout.count("[END]")
print(f"[START] count: {start_count}")
print(f"[END] count: {end_count}")
