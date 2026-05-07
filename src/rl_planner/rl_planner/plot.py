#!/usr/bin/env python3
"""Plot metrics for exploration or navigation.

Modes:
- exploration: legacy metrics_*.txt (4 columns)
- navigation: *_episodes.csv from eval_navigation.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_exploration_metrics(file_path: Path) -> np.ndarray:
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{file_path} has {data.shape[1]} columns; expected 4.")
    return data[:, :4]


def compute_completion_point(data: np.ndarray, distance_eps: float = 1e-6) -> Tuple[float, float]:
    distances = data[:, 1]
    times = data[:, 3]
    if len(distances) < 2:
        return float(times[-1]), float(distances[-1])
    diff = np.abs(np.diff(distances))
    change_indices = np.where(diff > distance_eps)[0]
    completion_idx = len(distances) - 1 if change_indices.size == 0 else int(change_indices[-1] + 1)
    return float(times[completion_idx]), float(distances[completion_idx])


def plot_exploration(metric_files: List[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    titles = [
        "Explored Volume (m^3)",
        "Traveling Distance (m)",
        "Algorithm Runtime (s)",
        "Completion (Time vs Distance)",
    ]
    ylabels = [
        "Explored Volume (m^3)",
        "Distance (m)",
        "Runtime (s)",
        "Total Distance (m)",
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, file_path in enumerate(metric_files):
        data = load_exploration_metrics(file_path)
        label = file_path.stem
        x = data[:, 3]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        for i in range(3):
            axes[i].plot(x, data[:, i], label=label, linewidth=1.5, color=color)
        completion_time, completion_distance = compute_completion_point(data)
        axes[3].scatter(
            completion_time,
            completion_distance,
            label=label,
            s=60,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel("Elapsed Time (s)" if i < 3 else "Completion Time (s)")
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.show()


def load_exploration_coverage_csv(file_path: Path) -> Dict[str, np.ndarray]:
    elapsed: List[float] = []
    covered_cells: List[float] = []
    total_cells: List[float] = []
    coverage_pct: List[float] = []
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed.append(float(row["elapsed_time_s"]))
            covered_cells.append(float(row["covered_free_cells"]))
            total_cells.append(float(row["total_free_cells"]))
            coverage_pct.append(float(row["coverage_percent"]))
    return {
        "elapsed_time_s": np.array(elapsed, dtype=float),
        "covered_free_cells": np.array(covered_cells, dtype=float),
        "total_free_cells": np.array(total_cells, dtype=float),
        "coverage_percent": np.array(coverage_pct, dtype=float),
    }


def load_exploration_summary(file_path: Path) -> Dict[str, float]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _find_summary_for_coverage(coverage_file: Path) -> Path | None:
    candidate = coverage_file.with_name(coverage_file.name.replace("_coverage.csv", "_summary.json"))
    return candidate if candidate.exists() else None


def plot_exploration_eval(coverage_files: List[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    labels: List[str] = []
    path_lengths: List[float] = []
    waypoint_hz: List[float] = []
    time70: List[float] = []
    time80: List[float] = []
    time90: List[float] = []

    for idx, fp in enumerate(coverage_files):
        data = load_exploration_coverage_csv(fp)
        label = fp.stem.replace("_coverage", "")
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        axes[0].plot(data["elapsed_time_s"], data["coverage_percent"], label=label, linewidth=1.6, color=color)
        axes[1].plot(data["elapsed_time_s"], data["covered_free_cells"], label=label, linewidth=1.6, color=color)

        summary_path = _find_summary_for_coverage(fp)
        if summary_path is not None:
            s = load_exploration_summary(summary_path)
            labels.append(label)
            path_lengths.append(float(s.get("path_length_m", 0.0)))
            waypoint_hz.append(float(s.get("waypoint_hz", 0.0)))
            time70.append(float(s["time_to_70"]) if s.get("time_to_70") is not None else np.nan)
            time80.append(float(s["time_to_80"]) if s.get("time_to_80") is not None else np.nan)
            time90.append(float(s["time_to_90"]) if s.get("time_to_90") is not None else np.nan)

    axes[0].set_title("Coverage (%) over Time")
    axes[0].set_xlabel("Elapsed Time (s)")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Covered Free Cells over Time")
    axes[1].set_xlabel("Elapsed Time (s)")
    axes[1].set_ylabel("Covered Free Cells")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    if labels:
        x = np.arange(len(labels))
        axes[2].bar(x - 0.18, path_lengths, width=0.36, label="Path Length (m)")
        axes[2].bar(x + 0.18, waypoint_hz, width=0.36, label="Waypoint Hz")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=15, ha="right")
        axes[2].set_title("Path Cost / Planning Frequency")
        axes[2].legend(frameon=False)
        axes[2].grid(True, linestyle="--", alpha=0.35, axis="y")

        w = 0.22
        axes[3].bar(x - w, time70, width=w, label="T@70%")
        axes[3].bar(x, time80, width=w, label="T@80%")
        axes[3].bar(x + w, time90, width=w, label="T@90%")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(labels, rotation=15, ha="right")
        axes[3].set_title("Time-to-Coverage Thresholds")
        axes[3].set_ylabel("Time (s)")
        axes[3].legend(frameon=False)
        axes[3].grid(True, linestyle="--", alpha=0.35, axis="y")
    else:
        axes[2].set_visible(False)
        axes[3].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.show()


def load_navigation_episodes(file_path: Path) -> Dict[str, List[float]]:
    cols = {
        "success": [],
        "success_time_s": [],
        "path_length_m": [],
        "final_goal_error_m": [],
        "waypoint_oscillation_count": [],
    }
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["success"].append(float(row["success"]))
            cols["path_length_m"].append(float(row["path_length_m"]))
            cols["final_goal_error_m"].append(float(row["final_goal_error_m"]))
            cols["waypoint_oscillation_count"].append(float(row["waypoint_oscillation_count"]))
            v = row["success_time_s"].strip()
            cols["success_time_s"].append(float(v) if v else np.nan)
    return cols


def plot_navigation(csv_files: List[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    titles = ["Success Rate (%)", "Time-to-Goal (s)", "Path Length (m)", "Final Error / Oscillation"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    labels = []
    success_rates = []
    mean_ttg = []
    mean_path = []
    mean_err = []
    mean_osc = []
    for fp in csv_files:
        d = load_navigation_episodes(fp)
        labels.append(fp.stem.replace("_episodes", ""))
        success_rates.append(100.0 * np.mean(d["success"]) if d["success"] else 0.0)
        ttg = np.array(d["success_time_s"], dtype=float)
        mean_ttg.append(float(np.nanmean(ttg)) if np.any(~np.isnan(ttg)) else np.nan)
        mean_path.append(float(np.mean(d["path_length_m"])) if d["path_length_m"] else 0.0)
        mean_err.append(float(np.mean(d["final_goal_error_m"])) if d["final_goal_error_m"] else 0.0)
        mean_osc.append(float(np.mean(d["waypoint_oscillation_count"])) if d["waypoint_oscillation_count"] else 0.0)

    x = np.arange(len(labels))
    colors = [color_cycle[i % len(color_cycle)] if color_cycle else None for i in range(len(labels))]
    axes[0].bar(x, success_rates, color=colors)
    axes[1].bar(x, mean_ttg, color=colors)
    axes[2].bar(x, mean_path, color=colors)
    w = 0.35
    axes[3].bar(x - w / 2, mean_err, width=w, label="Final Error (m)")
    axes[3].bar(x + w / 2, mean_osc, width=w, label="Oscillation Count")
    axes[3].legend(frameon=False)

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exploration", "navigation"], default="exploration")
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if args.mode == "exploration":
        coverage_files = sorted(input_dir.glob("*_coverage.csv"))
        if coverage_files:
            out = Path(args.output) if args.output else input_dir / "exploration_comparison.png"
            plot_exploration_eval(coverage_files, out)
            print(f"[OK] wrote {out}")
        else:
            files = sorted(input_dir.glob("metrics_*.txt"))
            if not files:
                raise SystemExit(f"No *_coverage.csv or metrics_*.txt found in {input_dir}")
            out = Path(args.output) if args.output else input_dir / "metrics_comparison.png"
            plot_exploration(files, out)
            print(f"[OK] wrote {out}")
    else:
        files = sorted(input_dir.glob("*_episodes.csv"))
        if not files:
            raise SystemExit(f"No *_episodes.csv found in {input_dir}")
        out = Path(args.output) if args.output else input_dir / "navigation_comparison.png"
        plot_navigation(files, out)
        print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
