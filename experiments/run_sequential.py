"""Sequential launcher v4 — all experiments use per-step subprocess isolation.
Exp2: done (skip). Exp3: quantize + eval separate. Exp4: single process (in-place swap).
"""
import subprocess, sys, time, os

PYTHON = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
ROOT = r"D:\Project\ptq-benchmark"
LOG_FILE = os.path.join(ROOT, "experiments", "sequential_run.log")

def run_cmd(args, desc, timeout=21600):
    cmd = [PYTHON] + args
    print(f"  [{desc}] {args[-2:] if len(args) > 2 else args}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    ok = result.returncode == 0
    return ok, result.stdout, result.stderr, result.returncode

def log(text):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()

def run_step(args, desc, timeout=21600):
    ok, out, err, code = run_cmd(args, desc, timeout=timeout)
    log(f"--- {desc} ---\nstdout:\n{out}")
    if err:
        log(f"stderr:\n{err}")
    log(f"exit: {code}")
    return ok

log(f"\n{'#'*60}")
log(f"Sequential v4 started at {time.ctime()}")
log(f"{'#'*60}")

# ---- Experiment 3: SmolLM config_b (quantize + eval separate) ----
EXP3 = "experiments/exp13_smollm_config_b/run.py"
log(f"\n{'='*60}")
log(f"Experiment 3: SmolLM-1.7B config_b (split quantize/eval)")
log(f"{'='*60}")

# Legacy non-compact state is known unsafe in this Windows/PyTorch setup.
# exp13 now writes results/SmolLM-1.7B_config_b_compact.pt, so keep the old
# artifact for provenance and avoid loading it.
smol_state = os.path.join(ROOT, "results/SmolLM-1.7B_config_b.pt")
if os.path.exists(smol_state):
    log("Legacy SmolLM-1.7B_config_b.pt exists; keeping it but using compact state path")

if not run_step([EXP3, "quantize"], "exp3-quantize", timeout=7200):
    log("*** exp3 quantize FAILED ***")
else:
    if not run_step([EXP3, "eval"], "exp3-eval", timeout=14400):
        log("*** exp3 eval FAILED ***")

# ---- Experiment 4: Layer Replacement (single process, in-place swap) ----
EXP4 = "experiments/exp07_layer_replacement/run.py"
log(f"\n{'='*60}")
log(f"Experiment 4: Layer Replacement (in-place swap, single load)")
log(f"{'='*60}")

# Clean up stale results
for f in ["layer_replacement.jsonl", "layer_replacement.csv", "layer_replacement.png"]:
    p = os.path.join(ROOT, "experiments/exp07_layer_replacement", f)
    if os.path.exists(p):
        os.remove(p)
        log(f"Deleted stale {f}")
    else:
        log(f"No stale {f} to delete")

if not run_step([EXP4], "exp4", timeout=25200):  # 7 hours for 26 evals
    log("*** exp4 FAILED ***")

log(f"\nSequential v4 finished at {time.ctime()}")
print("\nAll done.")
