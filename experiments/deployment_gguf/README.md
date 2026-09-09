# Frozen GGUF/CUDA deployment-transfer extension

This directory implements `deployment-gguf-v1`. It asks whether the **already
frozen SG-MMP allocation** transfers to a useful accuracy-memory-speed trade-off
on one packed backend. It does not replace the completed `revision-full-v4`
experiment and does not turn the original Python `GPTQLinear` into a deployment
kernel.

The scientific contract is in `protocol_lock.json`. In particular:

- every artifact starts from the immutable high-precision checkpoint;
- one pinned llama.cpp commit and one shared WikiText-train importance matrix are
  used within each model;
- the comparisons are `GGUF-FP16`, `Q4_0-pure`, `Q5_0-pure`, and
  `SG allocation-GGUF` (Q4_0 plus exact frozen Q8_0 tensor overrides);
- 32-element-block Q4_0/Q5_0 are intentional: the engineering audit found
  that K-quants structurally fall back on eligible Qwen tensor dimensions;
  no such fallback is accepted by the final per-tensor audit;
- embeddings and a stored output head remain F16;
- FP16 conversion fidelity uses identical-token-history, teacher-forced logits;
  free-running continuation equality is retained as a diagnostic because a
  numerically near-tied first token can amplify into a different trajectory;
- conversion evidence is bound to the locked gate-policy hash, and superseded
  gate records are archived instead of overwritten;
- no online `Question:` stopping is used; quality retains the original 5-shot,
  greedy, 256-token protocol;
- engineering, value-pilot, and formal process blocks live in different paths;
- test evaluation is impossible through the CLI until conversion, artifact, and
  CPU-vs-CUDA packed-backend gates have passed.

The high-precision conversion gate checks exact tokenizer IDs, greedy
continuations, and a frozen top-8 first-token log-probability tolerance against
the Hugging Face FP16 source. This separates conversion correctness from later
task-quality differences.

## What is measured

The artifact audit writes a row for every stored tensor, including its original
HF module (when eligible), GGUF name, shape, type, parameter count, payload
bytes, selection status, and tied-output status. It reports four quantities
separately: the old logical GPTQ bit budget, packed eligible bits/weight, whole
stored payload bits/element, and exact GGUF file bytes.

`llama-bench` measures prompt processing and token generation only; its numbers
are never labeled TTFT or end-to-end request throughput. The separate streaming
`llama-server` benchmark measures concurrency 1/4 TTFT, inter-token latency,
total request latency, aggregate throughput, model-load time, GPU memory, and
host RSS. Performance generation is fixed at 128 tokens with EOS ignored and is
not quality evidence.

Five within-process repetitions are summarized inside each process block. The
pilot uses five independent blocks. A formal run begins from a fresh `formal`
block namespace and uses at least ten blocks; analysis pairs methods by block
and bootstraps the geometric mean of block-level ratios. It never treats the
five repetitions as independent observations.

## Server prerequisites

The completed `revision-full-v4` selections, model snapshot, dataset snapshot,
Arrow caches, and local high-precision model directories must still be present.
Do not copy the new outputs into `experiments/revision_full/outputs`.

Build the exact backend once. The bootstrap deliberately creates a separate
Python environment next to (not inside) the llama.cpp checkout for the
converter, so its requirements cannot downgrade the established `LQ-sgmmp`
environment or make the pinned checkout appear dirty:

The host must provide CMake, Ninja/Make, `python3-venv`, CUDA Toolkit 12.4 at
`/usr/local/cuda-12.4`, and GCC/G++ 12. The paths are frozen because an earlier
CUDA 11.6 and 12.4 diagnostic builds showed that the pinned generic backend-op
suite has a randomized fused-F16 ADD case at its numerical tolerance boundary.
Those diagnostic attempts and logs are retained, but the randomized generic-op
suite is not an experimental eligibility gate and no post-hoc tolerance is
introduced. The pre-artifact backend gate requires the pinned build and the
quantization-function self-test to pass strictly. The FP16 conversion must pass
the frozen HF-versus-GGUF token and log-probability checks, and every produced
FP16, Q4, Q5, and SG artifact must pass the separate train-only CPU-versus-CUDA
continuation gate before benchmarking or test evaluation.
Compilation defaults to four parallel jobs to fit the 32-GiB server;
`DEPLOYMENT_BUILD_JOBS` may be raised only after observing adequate free RAM.

```bash
cd /data/experiment/LQ/sg-mmp-reproducibility
export DEPLOYMENT_LLAMA_CPP_DIR=/data/experiment/LQ/llama.cpp-deployment-gguf-v1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
bash experiments/deployment_gguf/bootstrap_llama_cpp.sh \
  2>&1 | tee logs/deployment_gguf_bootstrap.log
```

The experiment plan itself unsets network proxies and fails unless the checkout
is clean at commit `050dde50c9d70cf207db84f7224eedc491d817b2`, both RTX 3090-class
GPUs are visible, the CUDA binaries exist, and at least 40 GiB is free.
The imatrix builder uses the documented output frequency of 10 chunks; zero is
valid only for the snapshot-save frequency. It writes an incomplete GGUF first
and atomically publishes it after successful completion so a crash cannot leave
a resumable-looking final artifact.

## Stage 1: engineering pilot (Qwen-0.5B, no test data)

Generate, syntax-check, and launch the stage in tmux. GPU 1 is reserved for
timing because it normally has no desktop allocation; GPU 0 is reserved for
conversion/packed gates and later quality evaluation. The generator rejects a
plan that assigns both roles to the same GPU. Stop any other GPU job first.

```bash
cd /data/experiment/LQ/sg-mmp-reproducibility
conda activate LQ-sgmmp
source ./server_env.sh

python -m experiments.deployment_gguf.make_server_plan \
  --stage engineering \
  --llama-cpp-dir /data/experiment/LQ/llama.cpp-deployment-gguf-v1 \
  --timing-gpu 1 \
  --quality-gpu 0 \
  --output server_plans/deployment_gguf_engineering.sh

bash -n server_plans/deployment_gguf_engineering.sh
tmux new-session -d -s deploy_gguf_engineering \
  "cd /data/experiment/LQ/sg-mmp-reproducibility && bash server_plans/deployment_gguf_engineering.sh 2>&1 | tee logs/deployment_gguf_engineering.log"
```

This stage converts/builds all four Qwen-0.5B artifacts, constructs the common
importance matrix, audits every tensor, runs train-only conversion and packed
gates, and executes the five-block performance pilot. It never loads GSM8K
test. A gate failure terminates the shell before later commands.

Progress can be checked without attaching to tmux:

```bash
tmux ls 2>&1 || true
tail -n 80 logs/deployment_gguf_engineering.log
python -m experiments.deployment_gguf.run readiness --stage engineering
nvidia-smi
```

Closing VS Code or the SSH connection does not stop the tmux job.

## Later stages are intentionally manual

There is no automatic transition to test-bearing experiments. After the
engineering outputs have been inspected and any engineering fix has caused the
whole final pilot to be rerun, generate the precommitted value stage explicitly:

```bash
python -m experiments.deployment_gguf.make_server_plan \
  --stage value \
  --llama-cpp-dir /data/experiment/LQ/llama.cpp-deployment-gguf-v1 \
  --timing-gpu 1 \
  --quality-gpu 0 \
  --output server_plans/deployment_gguf_value.sh
```

That stage covers Qwen-1.5B and SmolLM2 and includes full 1319-item quality
measurement. Its generated plan completes artifacts, gates, and performance
blocks for **both** models before the first test row is loaded. The formal plan
uses the same all-model test firewall. The formal stage is generated with
`--stage formal`; it uses a
fresh ten-block namespace for Qwen-0.5B and Gemma, and should be launched only
after confirming the predeclared 24 GPU-hour and 40 GiB expansion caps.

Formal readiness is false after block 10 when any of the four primary native
service-throughput ratios (SG/Q4 and SG/Q5 at concurrency 1 and 4) still has a
95% paired-block CI half-width above 2%. In that case generate blocks 10--19,
then, only if still requested, 20--29. These extension plans do not rebuild
artifacts or rerun quality:

```bash
python -m experiments.deployment_gguf.make_server_plan \
  --stage formal \
  --start-block 10 \
  --end-block 20 \
  --llama-cpp-dir /data/experiment/LQ/llama.cpp-deployment-gguf-v1 \
  --timing-gpu 1 \
  --quality-gpu 0 \
  --output server_plans/deployment_gguf_formal_blocks_10_19.sh
```

At 30 paired blocks the protocol stops and reports the achieved interval even
if the 2% target remains unmet; it does not continue sampling until a favorable
result appears.

## Outputs and recovery

All generated artifacts are under
`experiments/deployment_gguf/outputs/` (or `DEPLOYMENT_GGUF_OUTPUT_DIR`) and are
gitignored. Quality JSONL is append-and-fsync resume safe. Completed performance
block files are immutable: rerunning the same command skips a matching block and
hard-fails if the artifact hash differs. Partial block commands may simply be
rerun because a block is registered only after the process exits successfully.

The final JSON/Markdown analysis is under
`experiments/deployment_gguf/outputs/analysis/<phase>/`. Preserve all pilot
outputs, including failed gate records; do not select only favorable runs.

## Local tests

```bash
python -m unittest experiments.deployment_gguf.test_deployment_gguf -v
python -m compileall -q experiments/deployment_gguf
```
