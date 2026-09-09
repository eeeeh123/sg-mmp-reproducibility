"""Pinned llama-server process and standard-library HTTP client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiments.deployment_gguf.protocol import CONTEXT_TOKENS_PER_SLOT, CPU_THREADS


def _post_json(url: str, payload: dict, timeout: float = 3600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gpu_memory_mib(gpu: int) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id",
                str(gpu),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def process_rss_mib(pid: int) -> float | None:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    return None


class GpuMemorySampler:
    def __init__(self, gpu: int, interval: float = 0.1):
        self.gpu = gpu
        self.interval = interval
        self.values: list[float] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("GPU memory sampler cannot be started twice")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id",
                str(self.gpu),
                f"--loop-ms={max(50, int(self.interval * 1000))}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.thread = None
            raise

    def _run(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            if self.stop_event.is_set():
                break
            try:
                self.values.append(float(line.strip()))
            except ValueError:
                continue

    def stop(self) -> None:
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5)


class LlamaServer:
    """Own one isolated llama-server and record its resource envelope."""

    def __init__(
        self,
        binary: Path,
        model: Path,
        log_path: Path,
        *,
        gpu: int = 0,
        slots: int = 1,
        cuda: bool = True,
        flash_attn: str | None = None,
        context_per_slot: int = CONTEXT_TOKENS_PER_SLOT,
        startup_timeout: float = 300,
    ):
        self.binary = binary.resolve()
        self.model = model.resolve()
        self.log_path = log_path.resolve()
        self.gpu = int(gpu)
        self.slots = int(slots)
        self.cuda = bool(cuda)
        if flash_attn not in (None, "on", "off"):
            raise ValueError("flash_attn must be on or off")
        self.flash_attn = flash_attn or ("on" if self.cuda else "off")
        self.context_per_slot = int(context_per_slot)
        self.startup_timeout = float(startup_timeout)
        self.port = free_local_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen | None = None
        self.log_stream = None
        self.sampler = GpuMemorySampler(self.gpu)
        self.baseline_gpu_mib: float | None = None
        self.loaded_gpu_mib: float | None = None
        self.peak_gpu_mib: float | None = None
        self.loaded_rss_mib: float | None = None
        self.startup_seconds: float | None = None

    @property
    def command(self) -> list[str]:
        command = [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_per_slot * self.slots),
            "--parallel",
            str(self.slots),
            "--threads",
            str(CPU_THREADS),
            "--threads-batch",
            str(CPU_THREADS),
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--load-mode",
            "none",
            "--lazy-mode",
            "off",
            "--cont-batching",
            "--no-cache-prompt",
            "--no-webui",
            "--log-jsonl",
            "--n-gpu-layers",
            "all" if self.cuda else "0",
            "--flash-attn",
            self.flash_attn,
        ]
        if not self.cuda:
            command.extend(["--device", "none"])
        return command

    def __enter__(self):
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError(self.binary if not self.binary.is_file() else self.model)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_gpu_mib = gpu_memory_mib(self.gpu) if self.cuda else None
        env = os.environ.copy()
        if self.cuda:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        started = time.perf_counter()
        try:
            self.sampler.start()
            self.log_stream = self.log_path.open("a", encoding="utf-8")
            self.process = subprocess.Popen(
                self.command,
                stdout=self.log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            deadline = time.monotonic() + self.startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with {self.process.returncode}; inspect {self.log_path}"
                    )
                try:
                    if _get_json(f"{self.base_url}/health").get("status") == "ok":
                        self.startup_seconds = time.perf_counter() - started
                        self.loaded_gpu_mib = gpu_memory_mib(self.gpu) if self.cuda else None
                        self.loaded_rss_mib = process_rss_mib(self.process.pid)
                        return self
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    last_error = exc
                time.sleep(0.25)
            raise TimeoutError(
                f"llama-server was not ready after {self.startup_timeout}s: {last_error}; "
                f"inspect {self.log_path}"
            )
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.sampler.stop()
        if self.sampler.values:
            self.peak_gpu_mib = max(self.sampler.values)
        if self.log_stream is not None:
            self.log_stream.close()

    def tokenize(self, content: str, *, add_special: bool = False) -> list[int]:
        response = _post_json(
            f"{self.base_url}/tokenize",
            {
                "content": content,
                "add_special": bool(add_special),
                "parse_special": True,
                "with_pieces": False,
            },
        )
        return [int(token) for token in response["tokens"]]

    def complete(
        self,
        prompt: str | list[int],
        *,
        n_predict: int,
        ignore_eos: bool = False,
        seed: int = 20260908,
        temperature: float = 0.0,
        n_probs: int = 0,
    ) -> dict:
        return _post_json(
            f"{self.base_url}/completion",
            {
                "prompt": prompt,
                "n_predict": int(n_predict),
                "temperature": float(temperature),
                "n_probs": int(n_probs),
                "seed": int(seed),
                "ignore_eos": bool(ignore_eos),
                "cache_prompt": False,
                "parse_special": True,
                "return_tokens": True,
                "stream": False,
            },
        )

    def stream_complete(
        self,
        prompt: str | list[int],
        *,
        n_predict: int,
        seed: int,
    ) -> dict:
        payload = {
            "prompt": prompt,
            "n_predict": int(n_predict),
            "temperature": 0.0,
            "seed": int(seed),
            "ignore_eos": True,
            "cache_prompt": False,
            "parse_special": True,
            "return_tokens": True,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        submitted = time.perf_counter()
        arrivals: list[float] = []
        token_ids: list[int] = []
        final = None
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data: "):
                    continue
                chunk = json.loads(line[6:].decode("utf-8"))
                if chunk.get("content") or chunk.get("tokens"):
                    arrivals.append(time.perf_counter())
                    token_ids.extend(int(token) for token in chunk.get("tokens", []))
                if chunk.get("stop"):
                    final = chunk
        ended = time.perf_counter()
        if not arrivals:
            raise RuntimeError("Streaming response contained no generated token")
        if final is None:
            raise RuntimeError("Streaming response ended without a final stop record")
        count = int(final.get("tokens_predicted", len(token_ids) or len(arrivals)))
        return {
            "submitted_monotonic": submitted,
            "first_token_monotonic": arrivals[0],
            "ended_monotonic": ended,
            "ttft_seconds": arrivals[0] - submitted,
            "latency_seconds": ended - submitted,
            "inter_token_latency_seconds": (
                None if len(arrivals) < 2 else (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1)
            ),
            "arrival_count": len(arrivals),
            "tokens_predicted": count,
            "stop_type": final.get("stop_type"),
            "server_timings": final.get("timings"),
        }

    def resource_record(self) -> dict:
        def delta(value):
            if value is None or self.baseline_gpu_mib is None:
                return None
            return value - self.baseline_gpu_mib

        return {
            "startup_seconds": self.startup_seconds,
            "baseline_gpu_memory_mib": self.baseline_gpu_mib,
            "loaded_gpu_memory_mib": self.loaded_gpu_mib,
            "peak_gpu_memory_mib": self.peak_gpu_mib,
            "loaded_model_gpu_delta_mib": delta(self.loaded_gpu_mib),
            "peak_gpu_delta_mib": delta(self.peak_gpu_mib),
            "loaded_host_rss_mib": self.loaded_rss_mib,
            "command": self.command,
        }
