#!/usr/bin/env python
"""Measure wall time, process RSS, and GPU memory for model commands.

This is deliberately an external runner. It monitors the existing workflow
entry points without importing or changing the model implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise SystemExit(
        "benchmark_overhead.py requires psutil; install it in the model environment"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MI_BYTES = 1024.0 * 1024.0


def _process_ids(root_pid: int) -> set[int]:
    """Return the monitored process and all currently live descendants."""
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return set()
    processes = [root, *root.children(recursive=True)]
    return {process.pid for process in processes if process.is_running()}


def _rss_mb(pids: set[int]) -> float:
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / MI_BYTES


def _gpu_memory_mb(pids: set[int]) -> float | None:
    """Sum GPU memory used by monitored PIDs, or return None if unavailable."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None or not pids:
        return None
    command = [
        nvidia_smi,
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    total = 0.0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            pid = int(fields[1])
            memory = float(fields[2])
        except ValueError:
            continue
        if pid in pids:
            total += memory
    return total


def monitor_command(
    label: str,
    command: list[str],
    *,
    sample_interval: float,
    require_gpu: bool,
) -> dict:
    """Run one command and sample aggregate memory for its process tree."""
    if not command:
        raise ValueError("command must not be empty")
    started = time.perf_counter()
    process = subprocess.Popen(command)
    peak_rss = 0.0
    peak_gpu: float | None = None
    gpu_observed = False
    while process.poll() is None:
        pids = _process_ids(process.pid)
        peak_rss = max(peak_rss, _rss_mb(pids))
        gpu_memory = _gpu_memory_mb(pids)
        if gpu_memory is not None:
            gpu_observed = True
            peak_gpu = gpu_memory if peak_gpu is None else max(peak_gpu, gpu_memory)
        time.sleep(sample_interval)

    pids = _process_ids(process.pid)
    peak_rss = max(peak_rss, _rss_mb(pids))
    gpu_memory = _gpu_memory_mb(pids)
    if gpu_memory is not None:
        gpu_observed = True
        peak_gpu = gpu_memory if peak_gpu is None else max(peak_gpu, gpu_memory)
    wall_seconds = time.perf_counter() - started
    if require_gpu and not gpu_observed:
        raise RuntimeError(
            "nvidia-smi did not report memory for the monitored process; "
            "check CUDA visibility or omit --require-gpu"
        )
    return {
        "method": "sparse_st_fgw",
        "phase": label,
        "command": shlex.join(command),
        "wall_time_seconds": wall_seconds,
        "peak_gpu_memory_mb": peak_gpu,
        "peak_rss_mb": peak_rss,
        "exit_code": process.returncode,
    }


def _project_command(args: argparse.Namespace, script: str) -> list[str]:
    command = [args.python, str(PROJECT_ROOT / "scripts" / script)]
    command.extend(["--data-dir", args.data_dir])
    if script == "train_model.py":
        command.extend(["--checkpoint", args.checkpoint, "--device", args.device])
    else:
        command.extend(
            [
                "--checkpoint",
                args.checkpoint,
                "--results-dir",
                args.results_dir,
                "--device",
                args.device,
            ]
        )
        if args.ot_device:
            command.extend(["--ot-device", args.ot_device])
    return command


def _build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.command:
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        return [(args.label, command)]
    commands = []
    if args.mode in {"train", "full"}:
        commands.append(("train", _project_command(args, "train_model.py")))
    if args.mode in {"inference", "full"}:
        commands.append(("inference", _project_command(args, "run_model.py")))
    return commands


def _write_report(report: dict, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [
        "method",
        "phase",
        "wall_time_seconds",
        "peak_gpu_memory_mb",
        "peak_rss_mb",
        "exit_code",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["phases"]:
            writer.writerow({field: row.get(field) for field in fields})
        if report.get("total") is not None:
            writer.writerow({field: report["total"].get(field) for field in fields})
    return output, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark model wall time, peak GPU memory, and peak RSS."
    )
    parser.add_argument("--mode", choices=("train", "inference", "full"), default="full")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights" / "spatial_graphsage.pt"),
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results_benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ot-device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--label", default="command")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results_benchmark" / "overhead.json")
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="monitor an arbitrary command instead of --mode; put it last",
    )
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")
    commands = _build_commands(args)
    if not commands or not commands[0][1]:
        raise ValueError("provide a model mode or a command after --command")
    phases = []
    started_at = datetime.now(timezone.utc).isoformat()
    for label, command in commands:
        result = monitor_command(
            label,
            command,
            sample_interval=args.sample_interval,
            require_gpu=args.require_gpu,
        )
        phases.append(result)
        if result["exit_code"] != 0:
            break
    successful = all(row["exit_code"] == 0 for row in phases)
    total = None
    if len(phases) > 1:
        total = {
            "method": "sparse_st_fgw",
            "phase": "train+inference",
            "command": "sequential phases",
            "wall_time_seconds": sum(row["wall_time_seconds"] for row in phases),
            "peak_gpu_memory_mb": max(
                (row["peak_gpu_memory_mb"] for row in phases if row["peak_gpu_memory_mb"] is not None),
                default=None,
            ),
            "peak_rss_mb": max(row["peak_rss_mb"] for row in phases),
            "exit_code": 0 if successful else phases[-1]["exit_code"],
        }
    report = {
        "format": "sparse_st_fgw_overhead_v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "sample_interval_seconds": args.sample_interval,
        "gpu_memory_source": "nvidia-smi process accounting",
        "rss_source": "psutil process-tree RSS",
        "status": "completed" if successful else "failed",
        "phases": phases,
        "total": total,
    }
    json_path, csv_path = _write_report(report, args.output)
    print(json.dumps({"status": report["status"], "json": str(json_path), "csv": str(csv_path)}, indent=2))
    if not successful:
        raise SystemExit(phases[-1]["exit_code"])
    return report


if __name__ == "__main__":
    main()
