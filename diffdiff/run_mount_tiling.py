import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffdiff.pipeline import StableDiffusionXLDiffImg2ImgPipeline


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resize_to_max_side(image: Image.Image, max_side: int | None) -> Image.Image:
    image = image.convert("RGB")
    if max_side is None:
        return image

    width, height = image.size
    current_max_side = max(width, height)
    if current_max_side <= max_side:
        return image

    scale = max_side / current_max_side
    new_width = max(64, int(round(width * scale)))
    new_height = max(64, int(round(height * scale)))
    return image.resize((new_width, new_height), resample=Image.LANCZOS)


def center_crop_to_multiple(image: Image.Image, multiple: int = 64) -> Image.Image:
    width, height = image.size
    cropped_height = height // multiple * multiple
    cropped_width = width // multiple * multiple
    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(f"Image size {image.size} is too small for a {multiple}-pixel multiple crop.")
    return transforms.CenterCrop((cropped_height, cropped_width))(image.convert("RGB"))


def prepare_source_image(image: Image.Image, max_side: int | None, multiple: int = 64) -> Image.Image:
    return center_crop_to_multiple(resize_to_max_side(image, max_side=max_side), multiple=multiple)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = center_crop_to_multiple(image, multiple=64)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    return image.unsqueeze(0)


def expand_mask_horizontally(mask_image: Image.Image, pad_pixels: int) -> Image.Image:
    if pad_pixels <= 0:
        return mask_image

    width, height = mask_image.size
    expanded = Image.new("L", (width + 2 * pad_pixels, height))
    expanded.paste(mask_image, (pad_pixels, 0))

    left_strip = mask_image.crop((width - pad_pixels, 0, width, height))
    right_strip = mask_image.crop((0, 0, pad_pixels, height))
    expanded.paste(left_strip, (0, 0))
    expanded.paste(right_strip, (pad_pixels + width, 0))
    return expanded


def preprocess_map(
        mask_image: Image.Image,
        target_size: tuple[int, int],
        max_width: int,
        vae_scale_factor: int = 8,
) -> torch.Tensor:
    mask_image = mask_image.convert("L")
    mask_image = mask_image.resize(target_size, resample=Image.BILINEAR)
    mask_image = center_crop_to_multiple(mask_image.convert("RGB"), multiple=64).convert("L")
    mask_image = expand_mask_horizontally(mask_image, pad_pixels=max_width * vae_scale_factor)
    return transforms.ToTensor()(mask_image)


def configure_pipeline_memory(pipe, cpu_offload: str, attention_slice_size: str) -> None:
    if hasattr(pipe, "enable_attention_slicing"):
        slice_value = "auto" if attention_slice_size == "auto" else int(attention_slice_size)
        pipe.enable_attention_slicing(slice_value)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


def tile_horizontally(image: Image.Image, repeats: int = 2) -> Image.Image:
    tiled = Image.new("RGB", (image.width * repeats, image.height))
    for i in range(repeats):
        tiled.paste(image, (i * image.width, 0))
    return tiled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use diffdiff to make mount.jpg tile seamlessly along the X axis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "test_images" / "mount.jpg",
        help="Path to the source image to tile.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=REPO_ROOT / "images" / "mask.jpg",
        help="Path to the grayscale diffdiff mask image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "diffdiff_mount",
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.22,
        help="Img2img strength. Lower values preserve more of the source image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Total denoising steps.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=12,
        help="Wrapped latent border width for X-axis tiling.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="Resize the longest image side to this value before processing. Lower it to 512 if VRAM is tight.",
    )
    parser.add_argument(
        "--cpu-offload",
        choices=("sequential", "model", "none"),
        default="sequential",
        help="How aggressively to offload SDXL weights to CPU to save VRAM.",
    )
    parser.add_argument(
        "--attention-slicing",
        default="auto",
        help="Attention slicing mode. Use 'auto' or a small integer like 1 or 2 for lower VRAM.",
    )
    parser.add_argument(
        "--use-refiner",
        action="store_true",
        help="Opt in to the SDXL refiner stage. This uses much more VRAM than base-only mode.",
    )
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Backward-compatible alias that forces base-only mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The diffdiff pipeline in this workspace requires a CUDA-capable GPU.")

    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask image not found: {args.mask}")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.strength <= 0 or args.strength > 1:
        raise ValueError("--strength must be in the range (0, 1].")
    if args.steps * args.strength < 1:
        raise ValueError(
            "The current --steps and --strength would produce zero effective denoising steps. "
            "Increase --steps or use a larger --strength. Example: --steps 8 --strength 0.22"
        )

    torch_dtype = torch.float16
    use_refiner = args.use_refiner and not args.no_refiner
    prompt = (
        "preserve the original mountain photo, keep the rock shapes, ridge lines, natural colors and fine details, "
        "only make the image seamlessly tileable horizontally"
    )
    negative_prompt = [
        "blurry, blurred, low detail, oversmoothed, distorted, deformed, washed out colors, artifacts"
    ]

    print(f"Loading source image: {args.input}")
    source_pil = Image.open(args.input).convert("RGB")
    cropped_source = prepare_source_image(source_pil, max_side=args.max_side, multiple=64)
    source_tensor = preprocess_image(cropped_source)

    print(f"Loading mask image: {args.mask}")
    mask_tensor = preprocess_map(
        Image.open(args.mask),
        target_size=cropped_source.size,
        max_width=args.max_width,
    ).to("cuda")

    print(f"Prepared source size: {cropped_source.size[0]}x{cropped_source.size[1]}")
    print(f"CPU offload mode: {args.cpu_offload}")
    print(f"Refiner enabled: {use_refiner}")

    print("Loading SDXL base differential diffusion pipeline...")
    base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        variant="fp16",
        use_safetensors=True,
    )
    configure_pipeline_memory(base, cpu_offload=args.cpu_offload, attention_slice_size=args.attention_slicing)

    if not use_refiner:
        print("Running base pipeline only...")
        with torch.inference_mode():
            edited_image = base(
                prompt=[prompt],
                original_image=source_tensor,
                image=source_tensor,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                map=mask_tensor,
                num_inference_steps=args.steps,
                max_width=args.max_width,
            ).images[0]
    else:
        print("Running base pipeline (latent output)...")
        with torch.inference_mode():
            latent_image = base(
                prompt=[prompt],
                original_image=source_tensor,
                image=source_tensor,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                map=mask_tensor,
                num_inference_steps=args.steps,
                denoising_end=0.8,
                max_width=args.max_width,
                output_type="latent",
            ).images

        shared_text_encoder_2 = base.text_encoder_2
        shared_vae = base.vae
        if args.cpu_offload == "none":
            base.to("cpu")
        del base
        base = None
        cleanup_cuda()

        print("Loading SDXL refiner differential diffusion pipeline...")
        refiner = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=shared_text_encoder_2,
            vae=shared_vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16",
        )
        configure_pipeline_memory(refiner, cpu_offload=args.cpu_offload, attention_slice_size=args.attention_slicing)

        print("Running refiner pipeline...")
        with torch.inference_mode():
            edited_image = refiner(
                prompt=[prompt],
                original_image=source_tensor,
                image=latent_image,
                strength=1.0,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                map=mask_tensor,
                num_inference_steps=args.steps,
                denoising_start=0.8,
                max_width=args.max_width,
            ).images[0]

        del latent_image
        del refiner
        del shared_text_encoder_2
        del shared_vae
        cleanup_cuda()

    if base is not None:
        del base
        cleanup_cuda()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_preview = tile_horizontally(cropped_source, repeats=2)
    tiled_preview = tile_horizontally(edited_image, repeats=2)

    source_output = output_dir / "mount_orig_cropped.png"
    source_preview_output = output_dir / "mount_orig_preview_1x2.png"
    tiled_output = output_dir / "mount_tiled.png"
    tiled_preview_output = output_dir / "mount_tiled_preview_1x2.png"

    cropped_source.save(source_output)
    original_preview.save(source_preview_output)
    edited_image.save(tiled_output)
    tiled_preview.save(tiled_preview_output)

    print("Done!")
    print(f"Saved cropped source image to: {source_output}")
    print(f"Saved original 1x2 preview to: {source_preview_output}")
    print(f"Saved tiled image to: {tiled_output}")
    print(f"Saved tiled 1x2 preview to: {tiled_preview_output}")


if __name__ == "__main__":
    main()





