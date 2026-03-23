"""
Tiled Diffusion 实验测试脚本
用法:
    python test.py --experiment A          # 运行 max_width 消融全部
    python test.py --experiment A --index 1 # 只运行 A-1
    python test.py --experiment B          # 运行 max_replica_width 消融（仅 many2many）
    python test.py --experiment blend      # 运行 blend_mode 消融（固定 mw=16, mrw=5）
    python test.py --experiment sample --blend_mode weighted  # 使用加权融合 padding 覆盖
    python test.py --experiment sample     # 运行全部固定样例（默认参数）
    python test.py --experiment sample --index 3  # 只运行样例 3
    python test.py --experiment one2one    # one-to-one 拼接
    python test.py --experiment many2many  # many-to-many 拼接
"""

import argparse
import gc
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from latent_class import LatentClass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# ======================== 全局固定配置 ========================

NEGATIVE_PROMPT = "blured, ugly, deformed, disfigured, poor details, bad anatomy, pixelized, bad order"
INFERENCE_STEPS = 40
CFG_SCALE = 7.5
SCHEDULER = "ddpm"
HEIGHT = 512
WIDTH = 512
STRENGTH = 0.95
DEFAULT_MAX_WIDTH = 16
DEFAULT_MAX_REPLICA_WIDTH = 3

# ======================== 固定测试样例 ========================

SAMPLES = [
    {
        "name": "brick_wall",
        "prompt": "Red brick wall texture, seamless, high quality, detailed mortar lines, realistic",
        "seed": 151,
        "side_id": [1, 1, 2, 2],
        "side_dir": ["cw", "ccw", "cw", "ccw"],
        "desc": "砖墙纹理 (self-tiling 四边)",
    },
    {
        "name": "grass_field",
        "prompt": "Green grass field texture, top view, seamless, natural, high detail, lush",
        "seed": 42,
        "side_id": [1, 1, 2, 2],
        "side_dir": ["cw", "ccw", "cw", "ccw"],
        "desc": "草地纹理 (self-tiling 四边)",
    },
    {
        "name": "marble_stone",
        "prompt": "Marble stone surface texture, seamless, white and grey veins, polished, high resolution",
        "seed": 233,
        "side_id": [1, 1, 2, 2],
        "side_dir": ["cw", "ccw", "cw", "ccw"],
        "desc": "大理石纹理 (self-tiling 四边)",
    },
    {
        "name": "geometric_pattern",
        "prompt": "Abstract geometric pattern, colorful triangles and hexagons, seamless, flat design, vector art style",
        "seed": 777,
        "side_id": [1, 1, 2, 2],
        "side_dir": ["cw", "ccw", "cw", "ccw"],
        "desc": "几何图案 (self-tiling 四边)",
    },
    {
        "name": "landscape",
        "prompt": "Rolling hills landscape, green meadow, blue sky with clouds, warm sunlight, oil painting style",
        "seed": 1024,
        "side_id": [1, 1, None, None],
        "side_dir": ["cw", "ccw", None, None],
        "desc": "风景 (self-tiling 仅左右)",
    },
]

# ======================== 消融实验配置 ========================

ABLATION_MAX_WIDTH = [4, 8, 12, 16, 32, 64]
ABLATION_MAX_REPLICA_WIDTH = [0, 1, 3, 5, 8, 16]
ABLATION_BLEND_MODES = ["overwrite", "weighted"]
BLEND_ABLATION_MAX_WIDTH = 16
BLEND_ABLATION_MAX_REPLICA_WIDTH = 5

# many-to-many 消融测试样例（实验 B 重点）
MANY2MANY_SAMPLES = [
    {"name": "brick_wall_m2m",
     "prompt": "Red brick wall texture, seamless, high quality, detailed mortar lines, realistic",
     "seed": 151},
    {"name": "grass_field_m2m",
     "prompt": "Green grass field texture, top view, seamless, natural, high detail, lush",
     "seed": 42},
    {"name": "wooden_floor_m2m",
     "prompt": "Wooden floor planks texture, seamless, natural oak, high detail",
     "seed": 999},
]

# ======================== 工具函数 ========================


def get_output_dir(experiment_name):
    out_dir = os.path.join("outputs", experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_timestamped_record_path(experiment_name):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", f"records_{experiment_name}_{timestamp}.json")


def find_latest_record_path(experiment_name):
    pattern = os.path.join("outputs", f"records_{experiment_name}_*.json")
    candidates = glob.glob(pattern)
    legacy_path = os.path.join("outputs", f"records_{experiment_name}.json")
    if os.path.exists(legacy_path):
        candidates.append(legacy_path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def save_image(img_array, path):
    """保存 numpy 图像数组为 PNG"""
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def make_tiled_preview(img_array, mode="2x2"):
    """生成拼接预览图"""
    if mode == "2x2":
        row = np.concatenate([img_array, img_array], axis=1)
        return np.concatenate([row, row], axis=0)
    elif mode == "1x2":
        return np.concatenate([img_array, img_array], axis=1)
    elif mode == "2x1":
        return np.concatenate([img_array, img_array], axis=0)


def save_boundary_crop(img_array, path, border_width=64):
    """保存拼接边界局部放大图（取左右拼接中线附近区域）"""
    h, w, c = img_array.shape
    tiled = np.concatenate([img_array, img_array], axis=1)
    # 裁切拼接中线附近
    center_x = w
    crop = tiled[:, center_x - border_width:center_x + border_width, :]
    save_image(crop, path)


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def run_single_experiment(model, sample, max_width, max_replica_width, out_dir, exp_label,
                          evaluator=None, blend_mode="overwrite"):
    """运行单次实验并保存结果，返回记录字典"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lat = LatentClass(
        prompt=sample["prompt"],
        negative_prompt=NEGATIVE_PROMPT,
        side_id=sample["side_id"],
        side_dir=sample["side_dir"],
    )

    start_time = time.time()
    new_latents = model(
        latents_arr=[lat],
        negative_prompt=NEGATIVE_PROMPT,
        inference_steps=INFERENCE_STEPS,
        seed=sample["seed"],
        cfg_scale=CFG_SCALE,
        height=HEIGHT,
        width=WIDTH,
        max_width=max_width,
        max_replica_width=max_replica_width,
        strength=STRENGTH,
        blend_mode=blend_mode,
        device=device,
    )
    elapsed = time.time() - start_time
    vram = get_vram_mb()

    img = new_latents[0].image
    prefix = f"{exp_label}_{sample['name']}"

    # 保存原图
    save_image(img, os.path.join(out_dir, f"{prefix}.png"))

    # 保存拼接预览
    has_y = sample["side_id"][2] is not None or sample["side_id"][3] is not None
    has_x = sample["side_id"][0] is not None or sample["side_id"][1] is not None
    if has_x and has_y:
        preview = make_tiled_preview(img, "2x2")
    elif has_x:
        preview = make_tiled_preview(img, "1x2")
    elif has_y:
        preview = make_tiled_preview(img, "2x1")
    else:
        preview = img
    save_image(preview, os.path.join(out_dir, f"{prefix}_tiled.png"))

    # 保存边界局部放大
    if has_x:
        save_boundary_crop(img, os.path.join(out_dir, f"{prefix}_boundary.png"))

    # 评估指标
    clip_score = None
    tiling_x_score = None
    tiling_y_score = None
    if evaluator is not None:
        clip_score = evaluator.evaluate_image_text_alignment(img, sample["prompt"])
        if has_x:
            tiling_x_score = evaluator.evaluate_tiling(img, img, direction='x')
        if has_y:
            tiling_y_score = evaluator.evaluate_tiling(img, img, direction='y')

    record = {
        "experiment": exp_label,
        "sample": sample["name"],
        "prompt": sample["prompt"],
        "seed": sample["seed"],
        "max_width": max_width,
        "max_replica_width": max_replica_width,
        "blend_mode": blend_mode,
        "inference_steps": INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "scheduler": SCHEDULER,
        "height": HEIGHT,
        "width": WIDTH,
        "time_seconds": round(elapsed, 2),
        "vram_mb": round(vram, 1),
        "clip_score": float(round(clip_score, 4)) if clip_score is not None else None,
        "tiling_x_mag": float(round(tiling_x_score, 6)) if tiling_x_score is not None else None,
        "tiling_y_mag": float(round(tiling_y_score, 6)) if tiling_y_score is not None else None,
    }

    metrics_str = f"耗时 {elapsed:.1f}s  |  显存 {vram:.0f}MB"
    if clip_score is not None:
        metrics_str += f"  |  CLIP={clip_score:.4f}"
    if tiling_x_score is not None:
        metrics_str += f"  |  MAG_x={tiling_x_score:.6f}"
    if tiling_y_score is not None:
        metrics_str += f"  |  MAG_y={tiling_y_score:.6f}"
    print(f"  [{exp_label}] {sample['name']}  |  {metrics_str}")
    return record


def run_multi_tile_experiment(model, experiment_type, max_width, max_replica_width, evaluator=None,
                              custom_out_dir=None, label_prefix=None,
                              prompt_override=None, seed_override=None, sample_name=None,
                              blend_mode="overwrite"):
    """运行多 tile 拼接实验"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = custom_out_dir or get_output_dir(experiment_type)

    if experiment_type == "one2one":
        prompt = prompt_override or "Medieval castle wall texture, stone blocks, seamless, detailed"
        seed = seed_override or 512
        lat_a = LatentClass(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            side_id=[1, None, None, None], side_dir=["cw", None, None, None],
        )
        lat_b = LatentClass(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            side_id=[None, 1, None, None], side_dir=[None, "ccw", None, None],
        )
        latents_arr = [lat_a, lat_b]
        name_tag = sample_name or "one2one"
        label = f"{label_prefix}_{name_tag}" if label_prefix else name_tag

    elif experiment_type == "many2many":
        prompt = prompt_override or "Wooden floor planks texture, seamless, natural oak, high detail"
        seed = seed_override or 999
        # 2x2: tile 0(右-1,下-2), tile 1(左-1,下-3), tile 2(右-4,上-2), tile 3(左-4,上-3)
        lat0 = LatentClass(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                           side_id=[1, None, None, 2], side_dir=["cw", None, None, "cw"])
        lat1 = LatentClass(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                           side_id=[None, 1, None, 3], side_dir=[None, "ccw", None, "cw"])
        lat2 = LatentClass(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                           side_id=[4, None, 2, None], side_dir=["cw", None, "ccw", None])
        lat3 = LatentClass(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                           side_id=[None, 4, 3, None], side_dir=[None, "ccw", "ccw", None])
        latents_arr = [lat0, lat1, lat2, lat3]
        name_tag = sample_name or "wooden_floor_m2m"
        label = f"{label_prefix}_{name_tag}" if label_prefix else name_tag
    else:
        return

    start_time = time.time()
    new_latents = model(
        latents_arr=latents_arr,
        negative_prompt=NEGATIVE_PROMPT,
        inference_steps=INFERENCE_STEPS,
        seed=seed,
        cfg_scale=CFG_SCALE,
        height=HEIGHT,
        width=WIDTH,
        max_width=max_width,
        max_replica_width=max_replica_width,
        strength=STRENGTH,
        blend_mode=blend_mode,
        device=device,
    )
    elapsed = time.time() - start_time
    vram = get_vram_mb()

    for i, lat in enumerate(new_latents):
        save_image(lat.image, os.path.join(out_dir, f"{label}_tile{i}.png"))

    # 拼接预览
    if experiment_type == "one2one":
        combined = np.concatenate([new_latents[0].image, new_latents[1].image], axis=1)
        save_image(combined, os.path.join(out_dir, f"{label}_combined.png"))
    elif experiment_type == "many2many":
        row0 = np.concatenate([new_latents[0].image, new_latents[1].image], axis=1)
        row1 = np.concatenate([new_latents[2].image, new_latents[3].image], axis=1)
        combined = np.concatenate([row0, row1], axis=0)
        save_image(combined, os.path.join(out_dir, f"{label}_combined.png"))

    # 评估指标
    clip_scores = []
    tiling_x_scores = []
    tiling_y_scores = []
    boundary_ssim_scores = []
    clip_consistency = None

    if evaluator is not None:
        for lat in new_latents:
            clip_scores.append(evaluator.evaluate_image_text_alignment(lat.image, prompt))

        if experiment_type == "one2one":
            tiling_x_scores.append(evaluator.evaluate_tiling(
                new_latents[0].image, new_latents[1].image, direction='x'))
            boundary_ssim_scores.append(evaluator.evaluate_boundary_ssim(
                new_latents[0].image, new_latents[1].image, direction='x'))

        elif experiment_type == "many2many":
            # 水平接缝: tile0-tile1, tile2-tile3
            tiling_x_scores.append(evaluator.evaluate_tiling(
                new_latents[0].image, new_latents[1].image, direction='x'))
            tiling_x_scores.append(evaluator.evaluate_tiling(
                new_latents[2].image, new_latents[3].image, direction='x'))
            # 垂直接缝: tile0-tile2, tile1-tile3
            tiling_y_scores.append(evaluator.evaluate_tiling(
                new_latents[0].image, new_latents[2].image, direction='y'))
            tiling_y_scores.append(evaluator.evaluate_tiling(
                new_latents[1].image, new_latents[3].image, direction='y'))
            # 边界 SSIM (四条接缝)
            boundary_ssim_scores.append(evaluator.evaluate_boundary_ssim(
                new_latents[0].image, new_latents[1].image, direction='x'))
            boundary_ssim_scores.append(evaluator.evaluate_boundary_ssim(
                new_latents[2].image, new_latents[3].image, direction='x'))
            boundary_ssim_scores.append(evaluator.evaluate_boundary_ssim(
                new_latents[0].image, new_latents[2].image, direction='y'))
            boundary_ssim_scores.append(evaluator.evaluate_boundary_ssim(
                new_latents[1].image, new_latents[3].image, direction='y'))

        # Tile 间视觉一致性
        tile_images = [lat.image for lat in new_latents]
        clip_consistency = evaluator.evaluate_clip_consistency(tile_images)

    valid_clip = [s for s in clip_scores if s is not None]
    avg_clip = float(round(np.mean(valid_clip), 4)) if valid_clip else None
    avg_mag_x = float(round(np.mean(tiling_x_scores), 6)) if tiling_x_scores else None
    avg_mag_y = float(round(np.mean(tiling_y_scores), 6)) if tiling_y_scores else None
    avg_bssim = float(round(np.mean(boundary_ssim_scores), 4)) if boundary_ssim_scores else None
    clip_cons = float(round(clip_consistency, 4)) if clip_consistency is not None else None

    metrics_str = f"耗时 {elapsed:.1f}s  |  显存 {vram:.0f}MB"
    if avg_clip is not None:
        metrics_str += f"  |  CLIP={avg_clip:.4f}"
    if avg_mag_x is not None:
        metrics_str += f"  |  MAG_x={avg_mag_x:.6f}"
    if avg_mag_y is not None:
        metrics_str += f"  |  MAG_y={avg_mag_y:.6f}"
    if avg_bssim is not None:
        metrics_str += f"  |  BSSIM={avg_bssim:.4f}"
    if clip_cons is not None:
        metrics_str += f"  |  CLIPcons={clip_cons:.4f}"
    print(f"  [{label}] 完成  |  {metrics_str}")

    return {
        "experiment": label,
        "sample": name_tag,
        "sample_type": experiment_type,
        "prompt": prompt,
        "seed": seed,
        "max_width": max_width,
        "max_replica_width": max_replica_width,
        "blend_mode": blend_mode,
        "time_seconds": round(elapsed, 2),
        "vram_mb": round(vram, 1),
        "clip_score": avg_clip,
        "tiling_x_mag": avg_mag_x,
        "tiling_y_mag": avg_mag_y,
        "boundary_ssim": avg_bssim,
        "clip_consistency": clip_cons,
    }


# ======================== 可视化与分析 ========================


def print_summary_table(records, ablation_key):
    """打印消融实验各参数组的指标汇总表"""
    from collections import defaultdict
    by_param = defaultdict(list)
    for r in records:
        by_param[r[ablation_key]].append(r)
    param_values = sorted(by_param.keys())
    print(f"\n{'='*78}")
    print(f"  Ablation Summary ({ablation_key})")
    print(f"{'='*78}")
    print(f"  {'Param':>8} | {'CLIP↑':>8} | {'MAG_x↓':>10} | {'MAG_y↓':>10} | {'Time(s)':>8} | {'VRAM(MB)':>9}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}")
    for pv in param_values:
        recs = by_param[pv]
        clips = [r['clip_score'] for r in recs if r.get('clip_score') is not None]
        mags_x = [r['tiling_x_mag'] for r in recs if r.get('tiling_x_mag') is not None]
        mags_y = [r['tiling_y_mag'] for r in recs if r.get('tiling_y_mag') is not None]
        times = [r['time_seconds'] for r in recs]
        vrams = [r['vram_mb'] for r in recs]
        c = f"{np.mean(clips):.4f}" if clips else "   N/A  "
        mx = f"{np.mean(mags_x):.6f}" if mags_x else "    N/A   "
        my = f"{np.mean(mags_y):.6f}" if mags_y else "    N/A   "
        t = f"{np.mean(times):.1f}"
        v = f"{np.mean(vrams):.0f}"
        print(f"  {pv:>8} | {c:>8} | {mx:>10} | {my:>10} | {t:>8} | {v:>9}")
    print(f"{'='*78}\n")


def visualize_ablation(records, ablation_key, out_dir):
    """生成消融实验指标对比图（CLIP / MAG / 资源消耗）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_param = defaultdict(list)
    for r in records:
        by_param[r[ablation_key]].append(r)
    param_values = sorted(by_param.keys())
    sample_names = list(dict.fromkeys(r['sample'] for r in records))
    colors = plt.cm.Set2(np.linspace(0, 1, len(sample_names)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _plot_metric(ax, key, title, ylabel):
        for idx, sample in enumerate(sample_names):
            pvs, vals = [], []
            for pv in param_values:
                rec = next((r for r in by_param[pv] if r['sample'] == sample), None)
                v = rec.get(key) if rec else None
                if v is not None:
                    pvs.append(pv)
                    vals.append(v)
            if vals:
                ax.plot(pvs, vals, 'o-', color=colors[idx], label=sample, alpha=0.7, markersize=6)
        avg_pvs, avg_vals = [], []
        for pv in param_values:
            vs = [r[key] for r in by_param[pv] if r.get(key) is not None]
            if vs:
                avg_pvs.append(pv)
                avg_vals.append(float(np.mean(vs)))
        if avg_vals:
            ax.plot(avg_pvs, avg_vals, 's--', color='black', linewidth=2.5, markersize=8,
                    label='Average', zorder=5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(ablation_key, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(param_values)

    _plot_metric(axes[0, 0], 'clip_score', 'CLIP Score (higher is better)', 'CLIP Score')
    _plot_metric(axes[0, 1], 'tiling_x_mag', 'MAG_x Tiling (lower is better)', 'MAG_x')
    _plot_metric(axes[1, 0], 'tiling_y_mag', 'MAG_y Tiling (lower is better)', 'MAG_y')

    # 资源消耗柱状图
    ax = axes[1, 1]
    avg_times = [float(np.mean([r['time_seconds'] for r in by_param[pv]])) for pv in param_values]
    avg_vrams = [float(np.mean([r['vram_mb'] for r in by_param[pv]])) for pv in param_values]
    x_pos = np.arange(len(param_values))
    w = 0.35
    ax2 = ax.twinx()
    ax.bar(x_pos - w / 2, avg_times, w, label='Time (s)', color='steelblue', alpha=0.7)
    ax2.bar(x_pos + w / 2, avg_vrams, w, label='VRAM (MB)', color='coral', alpha=0.7)
    ax.set_title('Resource Consumption', fontsize=12, fontweight='bold')
    ax.set_xlabel(ablation_key, fontsize=10)
    ax.set_ylabel('Time (s)', fontsize=10, color='steelblue')
    ax2.set_ylabel('VRAM (MB)', fontsize=10, color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_values)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Ablation Study: {ablation_key}', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(out_dir, f'ablation_{ablation_key}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  指标对比图已保存: {save_path}")


def visualize_boundary_ssim(records, ablation_key, out_dir):
    """生成 Boundary SSIM 指标图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    valid_records = [r for r in records if r.get('boundary_ssim') is not None]
    if not valid_records:
        return

    by_param = defaultdict(list)
    for record in valid_records:
        by_param[record[ablation_key]].append(record)

    param_values = sorted(by_param.keys())
    sample_names = list(dict.fromkeys(record['sample'] for record in valid_records))
    colors = plt.cm.Set2(np.linspace(0, 1, len(sample_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sample_name in enumerate(sample_names):
        x_values, y_values = [], []
        for param_value in param_values:
            record = next((item for item in by_param[param_value] if item['sample'] == sample_name), None)
            if record is not None:
                x_values.append(param_value)
                y_values.append(record['boundary_ssim'])
        if y_values:
            ax.plot(x_values, y_values, 'o-', color=colors[idx], label=sample_name, alpha=0.8, markersize=6)

    avg_values = [float(np.mean([record['boundary_ssim'] for record in by_param[param_value]])) for param_value in param_values]
    ax.plot(param_values, avg_values, 's--', color='black', linewidth=2.5, markersize=8, label='Average')
    ax.set_title('Boundary SSIM (higher is better)', fontsize=12, fontweight='bold')
    ax.set_xlabel(ablation_key, fontsize=10)
    ax.set_ylabel('Boundary SSIM', fontsize=10)
    ax.set_xticks(param_values)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    save_path = os.path.join(out_dir, f'boundary_ssim_{ablation_key}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Boundary SSIM 图已保存: {save_path}")


def create_comparison_grid(records, ablation_key, img_dir, out_dir):
    """生成不同参数设置下的图像对比网格"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_param = defaultdict(list)
    for r in records:
        by_param[r[ablation_key]].append(r)
    param_values = sorted(by_param.keys())
    sample_names = list(dict.fromkeys(r['sample'] for r in records))

    nrows, ncols = len(param_values), len(sample_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for row, pv in enumerate(param_values):
        for col, sample in enumerate(sample_names):
            ax = axes[row, col]
            rec = next((r for r in by_param[pv] if r['sample'] == sample), None)
            shown = False
            if rec:
                if rec.get("sample_type") == "many2many":
                    img_path = os.path.join(img_dir, f"{rec['experiment']}_combined.png")
                else:
                    img_path = os.path.join(img_dir, f"{rec['experiment']}_{sample}.png")
                if os.path.exists(img_path):
                    ax.imshow(Image.open(img_path))
                    shown = True
                    parts = []
                    if rec.get('clip_score') is not None:
                        parts.append(f"CLIP={rec['clip_score']:.3f}")
                    if rec.get('tiling_x_mag') is not None:
                        parts.append(f"MAG_x={rec['tiling_x_mag']:.4f}")
                    if parts:
                        ax.text(5, 25, "\n".join(parts), fontsize=7, color='white',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
            if not shown:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12,
                        color='gray', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(sample.replace('_', ' '), fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{ablation_key}={pv}', fontsize=9, fontweight='bold')

    fig.suptitle(f'Image Comparison ({ablation_key})', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(out_dir, f'comparison_grid_{ablation_key}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图像对比网格已保存: {save_path}")


def create_x_seam_grid(records, ablation_key, img_dir, out_dir, boundary_width=15):
    """显示完整的 x 方向 tiled 区域，并在图上标出 boundary 区域"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from matplotlib.patches import Rectangle

    by_param = defaultdict(list)
    for record in records:
        if record.get("tiling_x_mag") is not None:
            by_param[record[ablation_key]].append(record)

    if not by_param:
        print("  未找到可用于 x 方向接缝可视化的记录。")
        return

    param_values = sorted(by_param.keys())
    sample_names = list(dict.fromkeys(
        record["sample"] for record in records if record.get("tiling_x_mag") is not None
    ))

    nrows, ncols = len(param_values), len(sample_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 2.8 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for row, param_value in enumerate(param_values):
        for col, sample_name in enumerate(sample_names):
            ax = axes[row, col]
            record = next((item for item in by_param[param_value] if item["sample"] == sample_name), None)
            shown = False

            if record is not None:
                if record.get("sample_type") == "many2many":
                    tiled_path = os.path.join(img_dir, f"{record['experiment']}_combined.png")
                else:
                    tiled_path = os.path.join(img_dir, f"{record['experiment']}_{sample_name}_tiled.png")
                if os.path.exists(tiled_path):
                    tiled_image = np.array(Image.open(tiled_path).convert("RGB"))

                    # 1x2 tiled 图直接整张使用；2x2 tiled/combined 图取上半部分的横向拼接区域。
                    tiled_height, tiled_width, _ = tiled_image.shape
                    if tiled_width == tiled_height:
                        x_tiled_strip = tiled_image[: tiled_height // 2, :, :]
                    else:
                        x_tiled_strip = tiled_image

                    strip_height, strip_width, _ = x_tiled_strip.shape
                    center_x = strip_width // 2

                    ax.imshow(x_tiled_strip)

                    boundary_left = max(0, center_x - boundary_width * 4)
                    boundary_right = min(strip_width, center_x + boundary_width * 4)
                    rect = Rectangle(
                        (boundary_left, 0),
                        max(1, boundary_right - boundary_left),
                        strip_height,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.axvline(boundary_left, color="red", linestyle="--", linewidth=1)
                    ax.axvline(boundary_right, color="red", linestyle="--", linewidth=1)

                    info_lines = []
                    if record.get("tiling_x_mag") is not None:
                        info_lines.append(f"MAG_x={record['tiling_x_mag']:.4f}")
                    if record.get("clip_score") is not None:
                        info_lines.append(f"CLIP={record['clip_score']:.3f}")
                    if info_lines:
                        ax.text(
                            8,
                            20,
                            "\n".join(info_lines),
                            fontsize=7,
                            color="white",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
                        )

                    shown = True

            if not shown:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12,
                        color="gray", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(sample_name.replace("_", " "), fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{ablation_key}={param_value}", fontsize=9, fontweight="bold")

    fig.suptitle(f"X-direction Full Tiled View ({ablation_key})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(out_dir, f"x_seam_grid_{ablation_key}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  x 方向 tiled 接缝图已保存: {save_path}")


def _run_visualize():
    """从已保存的实验记录生成可视化分析"""
    found = False
    vis_dir = get_output_dir("visualize")
    for exp_name, ablation_key, dir_name in [
        ("A", "max_width", "ablation_max_width"),
        ("B", "max_replica_width", "ablation_max_replica_width"),
        ("blend", "blend_mode", "ablation_blend_mode"),
    ]:
        record_path = find_latest_record_path(exp_name)
        if not record_path:
            continue
        found = True
        with open(record_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        img_dir = os.path.join("outputs", dir_name)
        print(f"\n--- 可视化实验 {exp_name} ({ablation_key}) ---")
        print(f"  使用记录文件: {record_path}")
        print_summary_table(records, ablation_key)
        visualize_ablation(records, ablation_key, vis_dir)
        visualize_boundary_ssim(records, ablation_key, vis_dir)
        if os.path.isdir(img_dir):
            create_comparison_grid(records, ablation_key, img_dir, vis_dir)
            create_x_seam_grid(records, ablation_key, img_dir, vis_dir)
    if not found:
        print("未找到实验记录。请先运行实验 A、B 或 blend。")


# ======================== 主流程 ========================


def main():
    parser = argparse.ArgumentParser(description="Tiled Diffusion 实验脚本")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["sample", "A", "B", "blend", "one2one", "many2many", "visualize"],
                        help="实验类型: sample=固定样例, A=max_width消融, B=max_replica_width消融, blend=padding融合策略消融, one2one, many2many, visualize=可视化已有记录")
    parser.add_argument("--index", type=int, default=None,
                        help="指定样例编号(1-5)或消融编号(如A中1-6, B中1-6, blend中1-2)，不指定则全部运行")
    parser.add_argument("--max_width", type=int, default=None,
                        help="覆盖默认 max_width (实验B时使用)")
    parser.add_argument("--max_replica_width", type=int, default=None,
                        help="覆盖默认 max_replica_width")
    parser.add_argument("--blend_mode", type=str, default="overwrite",
                        choices=["overwrite", "weighted"],
                        help="padding 覆盖方式: overwrite=原始硬覆盖, weighted=线性加权融合")
    parser.add_argument("--no_eval", action="store_true",
                        help="跳过评估指标计算（加快运行速度）")
    parser.add_argument("--online", action="store_true",
                        help="允许从 Hugging Face 在线下载缺失模型文件")
    parser.add_argument("--hf_endpoint", type=str, default=None,
                        help="在线模式使用的 Hugging Face 端点，默认使用 https://hf-mirror.com")
    args = parser.parse_args()

    # 可视化模式不需要加载模型，直接从已有记录生成图表
    if args.experiment == "visualize":
        _run_visualize()
        return

    local_files_only = not args.online
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        hf_endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"在线模式端点: {hf_endpoint}")

    from model import SDLatentTiling
    from evaluator import Evaluator

    print(f"加载模型 (scheduler={SCHEDULER})...")
    try:
        model = SDLatentTiling(scheduler=SCHEDULER, local_files_only=local_files_only)
    except OSError as exc:
        if local_files_only:
            raise RuntimeError(
                "离线模式下未找到本地 Hugging Face 模型缓存。"
                "请先联网运行一次 `python test.py --experiment sample --index 1 --online --no_eval` 下载基础模型。"
            ) from exc
        raise

    evaluator = None
    if not args.no_eval:
        print("加载评估模型 (CLIP + MAG)...")
        try:
            evaluator = Evaluator(local_files_only=local_files_only)
        except OSError as exc:
            if local_files_only:
                raise RuntimeError(
                    "离线模式下未找到评估器所需的 Hugging Face 缓存。"
                    "请先联网运行一次 `python test.py --experiment sample --index 1 --online` 下载评估模型。"
                ) from exc
            raise

    all_records = []

    if args.experiment == "sample":
        out_dir = get_output_dir("samples")
        indices = [args.index - 1] if args.index else range(len(SAMPLES))
        for i in indices:
            s = SAMPLES[i]
            mw = args.max_width or DEFAULT_MAX_WIDTH
            mrw = args.max_replica_width or DEFAULT_MAX_REPLICA_WIDTH
            print(f"\n--- 样例 {i+1}: {s['desc']} ---")
            rec = run_single_experiment(
                model,
                s,
                mw,
                mrw,
                out_dir,
                f"S{i+1}",
                evaluator=evaluator,
                blend_mode=args.blend_mode,
            )
            all_records.append(rec)

    elif args.experiment == "A":
        out_dir = get_output_dir("ablation_max_width")
        mrw = args.max_replica_width or DEFAULT_MAX_REPLICA_WIDTH
        if args.index:
            widths = [ABLATION_MAX_WIDTH[args.index - 1]]
            labels = [f"A-{args.index}"]
        else:
            widths = ABLATION_MAX_WIDTH
            labels = [f"A-{j+1}" for j in range(len(widths))]
        for mw, label in zip(widths, labels):
            print(f"\n=== {label}: max_width={mw}, max_replica_width={mrw}, blend_mode={args.blend_mode} ===")
            for i, s in enumerate(SAMPLES):
                rec = run_single_experiment(
                    model,
                    s,
                    mw,
                    mrw,
                    out_dir,
                    label,
                    evaluator=evaluator,
                    blend_mode=args.blend_mode,
                )
                all_records.append(rec)

    elif args.experiment == "B":
        out_dir = get_output_dir("ablation_max_replica_width")
        mw = args.max_width or DEFAULT_MAX_WIDTH
        if args.index:
            rws = [ABLATION_MAX_REPLICA_WIDTH[args.index - 1]]
            labels = [f"B-{args.index}"]
        else:
            rws = ABLATION_MAX_REPLICA_WIDTH
            labels = [f"B-{j+1}" for j in range(len(rws))]
        for mrw, label in zip(rws, labels):
            print(f"\n=== {label}: max_width={mw}, max_replica_width={mrw}, blend_mode={args.blend_mode} (many2many only) ===")
            for m2m in MANY2MANY_SAMPLES:
                print(f"  --- {label}: many2many {m2m['name']} ---")
                rec_m2m = run_multi_tile_experiment(
                    model, "many2many", mw, mrw, evaluator=evaluator,
                    custom_out_dir=out_dir, label_prefix=label,
                    prompt_override=m2m['prompt'], seed_override=m2m['seed'],
                    sample_name=m2m['name'], blend_mode=args.blend_mode)
                if rec_m2m:
                    all_records.append(rec_m2m)

    elif args.experiment == "blend":
        out_dir = get_output_dir("ablation_blend_mode")
        mw = BLEND_ABLATION_MAX_WIDTH
        mrw = BLEND_ABLATION_MAX_REPLICA_WIDTH
        if args.index:
            blend_modes = [ABLATION_BLEND_MODES[args.index - 1]]
            labels = [f"C-{args.index}"]
        else:
            blend_modes = ABLATION_BLEND_MODES
            labels = [f"C-{j+1}" for j in range(len(blend_modes))]

        for blend_mode, label in zip(blend_modes, labels):
            print(f"\n=== {label}: max_width={mw}, max_replica_width={mrw}, blend_mode={blend_mode} ===")
            for sample in SAMPLES:
                rec = run_single_experiment(
                    model,
                    sample,
                    mw,
                    mrw,
                    out_dir,
                    label,
                    evaluator=evaluator,
                    blend_mode=blend_mode,
                )
                all_records.append(rec)

            for m2m in MANY2MANY_SAMPLES:
                print(f"  --- {label}: many2many {m2m['name']} ---")
                rec_m2m = run_multi_tile_experiment(
                    model,
                    "many2many",
                    mw,
                    mrw,
                    evaluator=evaluator,
                    custom_out_dir=out_dir,
                    label_prefix=label,
                    prompt_override=m2m['prompt'],
                    seed_override=m2m['seed'],
                    sample_name=m2m['name'],
                    blend_mode=blend_mode,
                )
                if rec_m2m:
                    all_records.append(rec_m2m)

    elif args.experiment in ("one2one", "many2many"):
        mw = args.max_width or DEFAULT_MAX_WIDTH
        mrw = args.max_replica_width or DEFAULT_MAX_REPLICA_WIDTH
        print(f"\n=== {args.experiment}: max_width={mw}, max_replica_width={mrw}, blend_mode={args.blend_mode} ===")
        rec = run_multi_tile_experiment(
            model,
            args.experiment,
            mw,
            mrw,
            evaluator=evaluator,
            blend_mode=args.blend_mode,
        )
        if rec:
            all_records.append(rec)

    # 保存实验记录
    if all_records:
        record_path = get_timestamped_record_path(args.experiment)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n实验记录已保存至 {record_path}")

        # 消融实验自动生成可视化
        if args.experiment == "A":
            ab_dir = get_output_dir("ablation_max_width")
            print_summary_table(all_records, 'max_width')
            visualize_ablation(all_records, 'max_width', ab_dir)
            visualize_boundary_ssim(all_records, 'max_width', ab_dir)
            create_comparison_grid(all_records, 'max_width', ab_dir, ab_dir)
            create_x_seam_grid(all_records, 'max_width', ab_dir, ab_dir)
        elif args.experiment == "B":
            ab_dir = get_output_dir("ablation_max_replica_width")
            print_summary_table(all_records, 'max_replica_width')
            visualize_ablation(all_records, 'max_replica_width', ab_dir)
            visualize_boundary_ssim(all_records, 'max_replica_width', ab_dir)
            create_comparison_grid(all_records, 'max_replica_width', ab_dir, ab_dir)
            create_x_seam_grid(all_records, 'max_replica_width', ab_dir, ab_dir)
        elif args.experiment == "blend":
            ab_dir = get_output_dir("ablation_blend_mode")
            print_summary_table(all_records, 'blend_mode')
            visualize_ablation(all_records, 'blend_mode', ab_dir)
            visualize_boundary_ssim(all_records, 'blend_mode', ab_dir)
            create_comparison_grid(all_records, 'blend_mode', ab_dir, ab_dir)
            create_x_seam_grid(all_records, 'blend_mode', ab_dir, ab_dir)


if __name__ == "__main__":
    main()
