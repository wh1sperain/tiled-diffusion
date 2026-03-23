import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_CONFIG = {
    "A": {
        "ablation_key": "max_width",
        "metrics": [
            ("clip_score", "CLIP Score", True),
            ("tiling_x_mag", "MAG_x", False),
            ("tiling_y_mag", "MAG_y", False),
            ("time_seconds", "Time (s)", False),
            ("vram_mb", "VRAM (MB)", False),
        ],
    },
    "B": {
        "ablation_key": "max_replica_width",
        "metrics": [
            ("clip_score", "CLIP Score", True),
            ("tiling_x_mag", "MAG_x", False),
            ("tiling_y_mag", "MAG_y", False),
            ("boundary_ssim", "Boundary SSIM", True),
            ("clip_consistency", "CLIP Consistency", True),
            ("time_seconds", "Time (s)", False),
            ("vram_mb", "VRAM (MB)", False),
        ],
    },
    "blend": {
        "ablation_key": "blend_mode",
        "metrics": [
            ("clip_score", "CLIP Score", True),
            ("tiling_x_mag", "MAG_x", False),
            ("tiling_y_mag", "MAG_y", False),
            ("boundary_ssim", "Boundary SSIM", True),
            ("clip_consistency", "CLIP Consistency", True),
            ("time_seconds", "Time (s)", False),
            ("vram_mb", "VRAM (MB)", False),
        ],
    },
}


def find_latest_record(experiment_name):
    pattern = os.path.join("outputs", f"records_{experiment_name}_*.json")
    candidates = glob.glob(pattern)
    legacy_path = os.path.join("outputs", f"records_{experiment_name}.json")
    if os.path.exists(legacy_path):
        candidates.append(legacy_path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_records(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def aggregate_means(records, ablation_key):
    grouped = {}
    for record in records:
        grouped.setdefault(record[ablation_key], []).append(record)

    params = sorted(grouped.keys())
    means = {}
    for param in params:
        means[param] = {}
        for key in records[0].keys():
            values = [item[key] for item in grouped[param] if item.get(key) is not None and isinstance(item.get(key), (int, float))]
            if values:
                means[param][key] = float(np.mean(values))
    return params, means


def plot_experiment_means(experiment_name, records, output_dir):
    config = EXPERIMENT_CONFIG[experiment_name]
    ablation_key = config["ablation_key"]
    params, means = aggregate_means(records, ablation_key)
    available_metrics = []

    for metric_key, title, higher_is_better in config["metrics"]:
        values = [means[param].get(metric_key) for param in params]
        if any(value is not None for value in values):
            available_metrics.append((metric_key, title, higher_is_better, values))

    if not available_metrics:
        print(f"实验 {experiment_name} 没有可绘制的均值指标。")
        return None

    cols = 3 if len(available_metrics) > 4 else 2
    rows = math.ceil(len(available_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for index, (metric_key, title, higher_is_better, values) in enumerate(available_metrics):
        row = index // cols
        col = index % cols
        ax = axes[row, col]
        y_values = [value if value is not None else np.nan for value in values]
        ax.plot(params, y_values, "o-", linewidth=2, markersize=7, color="#1f77b4")
        ax.set_title(f"{title} ({'higher' if higher_is_better else 'lower'} is better)")
        ax.set_xlabel(ablation_key)
        ax.set_ylabel(title)
        ax.set_xticks(params)
        ax.grid(True, alpha=0.3)

    for index in range(len(available_metrics), rows * cols):
        row = index // cols
        col = index % cols
        axes[row, col].axis("off")

    fig.suptitle(f"Experiment {experiment_name} Mean Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"experiment_{experiment_name}_mean_metrics.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="分析实验 JSON 记录并生成均值可视化图")
    parser.add_argument("--experiment", choices=["A", "B", "blend", "all"], default="all")
    parser.add_argument("--record-a", type=str, default=None, help="实验 A 的 JSON 路径，默认读取最新记录")
    parser.add_argument("--record-b", type=str, default=None, help="实验 B 的 JSON 路径，默认读取最新记录")
    parser.add_argument("--record-blend", type=str, default=None, help="实验 blend 的 JSON 路径，默认读取最新记录")
    parser.add_argument("--output-dir", type=str, default=os.path.join("outputs", "json_analysis"))
    args = parser.parse_args()

    experiments = [args.experiment] if args.experiment in {"A", "B", "blend"} else ["A", "B", "blend"]
    resolved_paths = {
        "A": args.record_a or find_latest_record("A"),
        "B": args.record_b or find_latest_record("B"),
        "blend": args.record_blend or find_latest_record("blend"),
    }

    for experiment_name in experiments:
        record_path = resolved_paths[experiment_name]
        if not record_path:
            print(f"未找到实验 {experiment_name} 的记录文件。")
            continue
        records = load_records(record_path)
        output_path = plot_experiment_means(experiment_name, records, args.output_dir)
        print(f"实验 {experiment_name} 使用记录: {record_path}")
        if output_path:
            print(f"实验 {experiment_name} 均值图已保存: {output_path}")


if __name__ == "__main__":
    main()