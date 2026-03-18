import gc
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from latent_class import LatentClass
from model import SDLatentTiling

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()
gc.collect()

scheduler = 'ddpm'
# scheduler = 'ddim'
model = SDLatentTiling(scheduler=scheduler)
# Parameters

prompt = "Red brick texture"
negative_prompt = "blured, ugly, deformed, disfigured, poor details, bad anatomy, pixelized, bad order"
inference_steps = 40
seed = 151
cfg_scale = 7.5
max_replica_width = 4
max_width = 32
height = 512
width = 512
input_image = None

######################### IMAGE TO IMAGE #########################
strength = 0.95
# input_image = Image.open("./test_images/mount.jpg").convert("RGB")
# input_image = input_image.resize((512, 512))
#################################################################

if input_image:
    width, height = input_image.size

# Right, Left, Up, Down
lat1 = LatentClass(prompt=prompt, negative_prompt=negative_prompt, side_id=[1, 1, 2, 2],
                   side_dir=['cw', 'ccw', 'cw', 'ccw'], source_image=input_image)

latents_arr = [lat1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
new_latents_arr = model(latents_arr=latents_arr,
                        negative_prompt=negative_prompt,
                        inference_steps=inference_steps,
                        seed=seed,
                        cfg_scale=cfg_scale,
                        height=height,
                        width=width,
                        max_width=max_width,
                        max_replica_width=max_replica_width,
                        strength=strength,
                        device=device)

torch.cuda.empty_cache()
gc.collect()

self_tiled_image = new_latents_arr[0].image
tiled_preview = np.concatenate((self_tiled_image, self_tiled_image), axis=1)

plt.figure(figsize=(12, 6))
plt.imshow(tiled_preview)
plt.axis('off')
plt.title('1x2 self-tiled txt2img preview')
plt.tight_layout()
plt.show()
