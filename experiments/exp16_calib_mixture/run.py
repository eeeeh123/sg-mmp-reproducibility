"""实验7: Calibration Mixture — 子进程隔离启动器。

每个 (method, ratio) 组合在独立 Python 子进程中运行，避免 WDDM 碎片化。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse, subprocess, time

N_SAMPLES = 128

RATIOS = {
    "100_0": (128, 0),
    "80_20": (102, 26),
    "60_40": (77, 51),
    "40_60": (51, 77),
    "20_80": (26, 102),
    "0_100": (0, 128),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated, e.g. 100_0,50_50,0_100")
    parser.add_argument("--methods", type=str, default="gptq,config_b",
                        help="Comma-separated: gptq,config_b")
    parser.add_argument("--skip_quantize", action="store_true")
    args = parser.parse_args()

    if args.ratios:
        selected = {}
        for r in args.ratios.split(","):
            wiki, gsm = r.split("_")
            selected[r] = (int(wiki) * 128 // 100, int(gsm) * 128 // 100)
    else:
        selected = RATIOS

    methods = [m.strip() for m in args.methods.split(",")]
    python_exe = sys.executable
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")

    total = len(methods) * len(selected)
    print(f"Experiment 7 Phase {args.phase}")
    print(f"  {len(methods)} methods x {len(selected)} ratios = {total} combinations")
    print(f"  Worker: {worker}")
    print()

    n = 0
    for method in methods:
        for ratio_tag, (n_wiki, n_gsm8k) in sorted(selected.items()):
            n += 1
            cmd = [
                python_exe, worker,
                "--method", method,
                "--ratio_tag", ratio_tag,
                "--n_wiki", str(n_wiki),
                "--n_gsm8k", str(n_gsm8k),
                "--phase", str(args.phase),
            ]
            if args.skip_quantize:
                cmd.append("--skip_quantize")

            print(f"[{n}/{total}] {method} @ {ratio_tag} (wiki={n_wiki}, gsm8k={n_gsm8k})")
            t0 = time.time()
            r = subprocess.run(cmd, cwd="D:/Project/ptq-benchmark",
                               capture_output=True, text=True)
            elapsed = time.time() - t0

            if r.returncode != 0:
                print(f"  FAILED (exit {r.returncode}, {elapsed:.0f}s)")
                print(f"  STDERR: {r.stderr[-200:]}")
                print(f"  STDOUT: {r.stdout[-200:]}")
            else:
                print(f"  OK ({elapsed:.0f}s)")
                # Show eval results from worker output
                for line in r.stdout.strip().splitlines():
                    line = line.strip()
                    if line and ("Eval:" in line or "Worker done" in line):
                        print(f"    {line}")
            print()
            time.sleep(2)  # Cool-down between subprocesses

    print(f"\nExperiment 7 Phase {args.phase} complete. {n} combinations attempted.")


if __name__ == "__main__":
    main()
