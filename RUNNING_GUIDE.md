# Tiled Diffusion — 快速运行指南

## 检查清单（运行前）
- [ ] 有可用 GPU（推荐 NVIDIA + CUDA），或准备接受 CPU 极慢的运行速度  
- [ ] 已安装 conda（或可使用系统 Python）  
- [ ] 已从仓库根目录运行命令（包含 `run.py` 的目录）  
- [ ] 如果需要，已登录 Hugging Face（见“模型认证”一节）

---

## 一、在 Windows (PowerShell) 上创建环境并安装依赖
在项目根目录（含 `requirements.txt`）中运行：

```powershell
conda create -n td python=3.10 -y
conda activate td
pip install --upgrade pip
pip install -r requirements.txt
```

注意：
- `requirements.txt` 中的 torch 版本为 CUDA 11.8（`+cu118`）。如果你的 CUDA 版本不同，请根据官方 PyTorch 安装说明选择合适的 wheel。
- 若模型受限或私有，请先登录 Hugging Face：`huggingface-cli login`

---

## 二、运行默认示例（最小命令）
在仓库根目录直接运行 `run.py`：

```powershell
python run.py
```

默认会：
- 加载 `SDLatentTiling`（在 `model.py`）并下载所需权重（若本地未缓存）
- 生成两个 `LatentClass`，进行 many-to-many X 轴 tiling 示例
- 最后使用 matplotlib 弹窗显示拼接图（如想保存为文件请参见下文）

关键代码位置：
- `run.py`（入口示例）
- `model.py`（模型加载与推理）
- `latent_class.py`（tile/边配置）

---

## 三、`run.py` 常用输入参数（及含义 / 默认示例）
这些参数在 `run.py` 顶部或 `LatentClass` 构造中设置：

- `scheduler`：采样器，示例 `'ddpm'`（可选 `'ddim'` 或 Euler）
- `prompt_1`, `prompt_2`：字符串，正向 prompt（示例:"Red brick texture" / "Green brick texture"）
- `negative_prompt`：字符串，负向提示（示例:"blured, ugly, ..."）
- `inference_steps`：int，采样步数（示例 40）——越大通常质量越高但越慢
- `seed`：int（示例 151），用于复现
- `cfg_scale`：float 指导尺度（示例 7.5）
- `max_width`：int（示例 32），latent padding 大小（影响无缝拼接与显存）
- `max_replica_width`：int（示例 5），相似性约束复制宽度上限
- `height`, `width`：输出像素（示例 512,512）——注意网络内部以 8 为缩放因子
- `strength`：float（示例 0.92），仅 img2img 有意义（越接近 1 表示更强的变换行为，可参考 pipeline 文档）
- `input_image` / `source_image`：若传入 `PIL.Image` 则走 img2img 分支
- `LatentClass` 的 `side_id` 与 `side_dir`（长度 4，顺序：[Right, Left, Up, Down]）
  - 示例 self-tiling：`side_id=[1,1,2,2]`, `side_dir=['cw','ccw','cw','ccw']`
  - many-to-many X 轴：`side_id=[1,1,None,None]`, `side_dir=['cw','ccw',None,None]`

默认（`run.py` 示例）摘要：
- scheduler = 'ddpm'
- inference_steps = 40
- seed = 151
- cfg_scale = 7.5
- max_replica_width = 5
- max_width = 32
- height = 512, width = 512
- strength = 0.92（img2img）

---

## 四、启用 img2img（在 `run.py`）
在 `run.py` 中有注释示例，取消注释并将 `input_image` 设置为 PIL 图像即可（示例为从 URL 下载）：

示例代码片段（已在 `run.py` 中）：
```python
from PIL import Image
from io import BytesIO
import requests

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
input_image = Image.open(BytesIO(response.content)).convert("RGB")
input_image = input_image.resize((768, 512))

# 使用 source_image 创建 LatentClass
lat1 = LatentClass(prompt=..., source_image=input_image, side_id=[...], side_dir=[...])
```

`strength` 参数会影响 img2img 的保真度/变换强度。

---

## 五、其它示例运行说明
- `controlnet/example.py`
  - 会使用 `lllyasviel/sd-controlnet-canny` 和 `runwayml/stable-diffusion-v1-5`。脚本会读取 `input` 目录下的 PNG，输出到 `res`, `init`, `mask` 等文件夹。运行：
    ```powershell
    python controlnet/example.py
    ```
  - 注意：确保 `input` 等目录存在或修改脚本中的路径。

- `sd3/example.py`
  - 使用 `stabilityai/stable-diffusion-3-medium-diffusers`，运行：
    ```powershell
    python sd3/example.py
    ```

- `diffdiff/example.py`（SDXL / Differential Diffusion）
  - 先用 text->image pipeline 生成 base image，再走 img2img/refiner 流程并保存若干 tiled 图像：
    ```powershell
    python diffdiff/example.py
    ```

- `sdxl/example.py`
  - 运行相应文件以测试 SDXL pipeline（若已按示例安装依赖、并有足够显存）：
    ```powershell
    python sdxl/example.py
    ```

---

## 六、模型文件在哪里？如何预下载 / 更改缓存位置
- 代码通过 `from_pretrained(...)`（Hugging Face API）在线下载模型并缓存到本地。仓库本身不包含权重。
- Windows 默认缓存目录（Hugging Face hub）通常为：
  ```
  C:\Users\<用户名>\.cache\huggingface\hub\
  ```
- 若想预下载模型（示例：`runwayml/stable-diffusion-v1-5`、`lllyasviel/sd-controlnet-canny`），可用一个小 Python 脚本（保存为 `download_models.py`）：

`download_models.py`
```python
from huggingface_hub import snapshot_download

models = [
    "runwayml/stable-diffusion-v1-5",
    "lllyasviel/sd-controlnet-canny",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

for m in models:
    print("Downloading:", m)
    path = snapshot_download(m)
    print("Saved to:", path)
```

然后在 PowerShell 运行：
```powershell
python download_models.py
```

- 更改缓存位置（临时在当前 shell 生效）：
```powershell
$env:HF_HOME = "D:\hf_cache"
$env:TRANSFORMERS_CACHE = "D:\transformers_cache"
# 然后运行 python 脚本，模型会下载到新的缓存目录
```

- 若模型私有或需要认证，请先登录：
```powershell
huggingface-cli login
```

或者在代码调用 `from_pretrained(..., use_auth_token=True)`。

---

## 七、保存输出（替换 `plt.show()`）
`run.py` 默认用 matplotlib 显示结果。如果你想保存成 PNG，修改代码末尾显示部分（示例替换片段）：

示例替换（把最后的 plt.show() 替换为保存）：
```python
import imageio
# t_1 = np.concatenate(...)
# 保存为 PNG
import numpy as np
from PIL import Image
img_uint8 = (t_1 * 255).astype('uint8')  # 如果 t_1 在 [0,1]
im = Image.fromarray(img_uint8)
im.save("output/tiled_result.png")
```

示例（若 `t_1` 已是 uint8）：
```python
from PIL import Image
Image.fromarray(t_1).save("output/tiled_result.png")
```

（注意：确保创建 `output` 文件夹或用绝对路径）

---

## 八、生成循环 GIF（`gif_creator.py`）
示例（从命令行）：
```powershell
python gif_creator.py path\to\images path\to\gifs right --target-size 512 --duration 5000 --frames 30
```

或在 Python 中调用：
```python
from gif_creator import process_directory
process_directory("generated_images", "gifs", "right", duration=5000, num_frames=30, target_size=512)
```

---

## 九、常见问题与调优建议
- 显存 OOM：
  - 降低 `height` / `width`
  - 减小 `max_width` / `max_replica_width`
  - 减少 `inference_steps`
  - 使用模型的 CPU offload（若 pipeline 支持），或在多次运行之间调用 `torch.cuda.empty_cache()` 和 `gc.collect()`
- CPU 上运行会非常慢；强烈建议使用 GPU
- 若 from_pretrained 报错（认证 / 403），请确认 HF token 或模型是否公开
- 若出现尺寸不对（UNet/vae 要求），使用能被 8 整除 / 常见的 64 的倍数尺寸

---

## 十、快速参考（常用命令集）
创建环境并安装：
```powershell
conda create -n td python=3.10 -y
conda activate td
pip install --upgrade pip
pip install -r requirements.txt
```

在项目根目录运行默认示例：
```powershell
python run.py
```

预下载模型（示例脚本）：
```powershell
python download_models.py
```

生成 GIF：
```powershell
python gif_creator.py generated_images gifs right --target-size 512 --duration 5000 --frames 30
```

---

如果你希望，我可以：
- 生成并返回一个 `download_models.py` 完整脚本（你可直接保存并运行）；
- 给出精确的 `run.py` 修改建议（把结果保存到文件夹），并贴出完整替换片段；
- 或者根据你的机器（是否有 GPU、CUDA 版本）给出更具体的依赖安装建议。

你想先要哪一个（生成 `download_models.py`、给出 `run.py` 保存输出的完整修改片段，还是按你的机器定制安装步骤）？

