"""
Tiled Diffusion 实验测试脚本
用法:
    python test.py --experiment A          # 运行 max_width 消融全部
    python test.py --experiment A --index 1 # 只运行 A-1
    python test.py --experiment B          # 运行 max_replica_width 消融全部
    python test.py --experiment sample     # 运行全部固定样例（默认参数）
    python test.py --experiment sample --index 3  # 只运行样例 3
    python test.py --experiment one2one    # one-to-one 拼接
    python test.py --experiment many2many  # many-to-many 拼接
"""

import argparse
import gc
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from latent_class import LatentClass
from model import SDLatentTiling

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

ABLATION_MAX_WIDTH = [4, 8, 12, 16, 32]
ABLATION_MAX_REPLICA_WIDTH = [1, 3, 5, 8]

# ======================== 工具函数 ========================


def get_output_dir(experiment_name):
    out_dir = os.path.join("outputs", experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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


def run_single_experiment(model, sample, max_width, max_replica_width, out_dir, exp_label):
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

    record = {
        "experiment": exp_label,
        "sample": sample["name"],
        "prompt": sample["prompt"],
        "seed": sample["seed"],
        "max_width": max_width,
        "max_replica_width": max_replica_width,
        "inference_steps": INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "scheduler": SCHEDULER,
        "height": HEIGHT,
        "width": WIDTH,
        "time_seconds": round(elapsed, 2),
        "vram_mb": round(vram, 1),
    }

    print(f"  [{exp_label}] {sample['name']}  |  耗时 {elapsed:.1f}s  |  显存 {vram:.0f}MB")
    return record


def run_multi_tile_experiment(model, experiment_type, max_width, max_replica_width):
    """运行多 tile 拼接实验"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = get_output_dir(experiment_type)

    if experiment_type == "one2one":
        prompt = "Medieval castle wall texture, stone blocks, seamless, detailed"
        seed = 512
        lat_a = LatentClass(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            side_id=[1, None, None, None], side_dir=["cw", None, None, None],
        )
        lat_b = LatentClass(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            side_id=[None, 1, None, None], side_dir=[None, "ccw", None, None],
        )
        latents_arr = [lat_a, lat_b]
        label = "one2one"

    elif experiment_type == "many2many":
        prompt = "Wooden floor planks texture, seamless, natural oak, high detail"
        seed = 999
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
        label = "many2many"
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

    print(f"  [{label}] 完成  |  耗时 {elapsed:.1f}s  |  显存 {vram:.0f}MB")
    return {
        "experiment": label,
        "seed": seed,
        "max_width": max_width,
        "max_replica_width": max_replica_width,
        "time_seconds": round(elapsed, 2),
        "vram_mb": round(vram, 1),
    }


# ======================== 主流程 ========================


def main():
    parser = argparse.ArgumentParser(description="Tiled Diffusion 实验脚本")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["sample", "A", "B", "one2one", "many2many"],
                        help="实验类型: sample=固定样例, A=max_width消融, B=max_replica_width消融, one2one, many2many")
    parser.add_argument("--index", type=int, default=None,
                        help="指定样例编号(1-5)或消融编号(如A中1-5, B中1-4)，不指定则全部运行")
    parser.add_argument("--max_width", type=int, default=None,
                        help="覆盖默认 max_width (实验B时使用)")
    parser.add_argument("--max_replica_width", type=int, default=None,
                        help="覆盖默认 max_replica_width")
    args = parser.parse_args()

    print(f"加载模型 (scheduler={SCHEDULER})...")
    model = SDLatentTiling(scheduler=SCHEDULER)
    all_records = []

    if args.experiment == "sample":
        out_dir = get_output_dir("samples")
        indices = [args.index - 1] if args.index else range(len(SAMPLES))
        for i in indices:
            s = SAMPLES[i]
            mw = args.max_width or DEFAULT_MAX_WIDTH
            mrw = args.max_replica_width or DEFAULT_MAX_REPLICA_WIDTH
            print(f"\n--- 样例 {i+1}: {s['desc']} ---")
            rec = run_single_experiment(model, s, mw, mrw, out_dir, f"S{i+1}")
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
            print(f"\n=== {label}: max_width={mw}, max_replica_width={mrw} ===")
            for i, s in enumerate(SAMPLES):
                rec = run_single_experiment(model, s, mw, mrw, out_dir, label)
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
            print(f"\n=== {label}: max_width={mw}, max_replica_width={mrw} ===")
            for i, s in enumerate(SAMPLES):
                rec = run_single_experiment(model, s, mw, mrw, out_dir, label)
                all_records.append(rec)

    elif args.experiment in ("one2one", "many2many"):
        mw = args.max_width or DEFAULT_MAX_WIDTH
        mrw = args.max_replica_width or DEFAULT_MAX_REPLICA_WIDTH
        print(f"\n=== {args.experiment}: max_width={mw}, max_replica_width={mrw} ===")
        rec = run_multi_tile_experiment(model, args.experiment, mw, mrw)
        if rec:
            all_records.append(rec)

    # 保存实验记录
    if all_records:
        record_path = os.path.join("outputs", f"records_{args.experiment}.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n实验记录已保存至 {record_path}")


if __name__ == "__main__":
    main()
