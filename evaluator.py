import lpips
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import warnings
import cv2
from PIL import Image
from clip import clip
from scipy.stats import wasserstein_distance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils import mean_absolute_gradient


class Evaluator:

    def __init__(self, local_files_only=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.clip_model = None
        self.clip_preprocess = None
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as exc:
            warnings.warn(f"OpenAI CLIP 加载失败，将跳过图文对齐指标: {exc}")
        self.inception_model = None
        self.fid_metric = None
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.fid_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.lpips_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loss_fn = None

    def _ensure_inception_model(self):
        if self.inception_model is None:
            self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
            self.inception_model.fc = torch.nn.Identity()
        if self.fid_metric is None:
            self.fid_metric = FrechetInceptionDistance(feature=64)

    def _ensure_lpips_model(self):
        if self.loss_fn is None:
            self.loss_fn = lpips.LPIPS(net='alex')

    def calculate_lpips(self, generated_image, ground_truth_image):
        # Preprocess images
        self._ensure_lpips_model()
        generated_tensor = self.lpips_transform(generated_image).unsqueeze(0)
        ground_truth_tensor = self.lpips_transform(ground_truth_image).unsqueeze(0)

        # Compute LPIPS distance
        with torch.no_grad():
            lpips_distance = self.loss_fn(generated_tensor, ground_truth_tensor)

        # Convert to scalar and return
        return lpips_distance.item()

    def calculate_fid(self, generated_image, ground_truth_image):
        # Load and preprocess images
        self._ensure_inception_model()

        # generated_image = Image.open(generated_image_path).convert('RGB')
        # ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')

        generated_tensor = self.fid_transform(generated_image).unsqueeze(0)
        ground_truth_tensor = self.fid_transform(ground_truth_image).unsqueeze(0)

        # Initialize FID metric
        fid = FrechetInceptionDistance(feature=64)

        # Update FID with images
        fid.update(generated_tensor, real=False)
        fid.update(ground_truth_tensor, real=True)

        # Calculate FID score
        fid_score = fid.compute().item()

        return fid_score

    def evaluate_image_text_alignment(self, image, prompt):
        """
        This evaluation method generates a cosine similarity scalar between the generated image and its prompt
        :param image: Generated image
        :param prompt: Prompt that from it the image was generated
        :return: Cosine similarity score between the image and the prompt based on clip encoding
        """
        if self.clip_model is None or self.clip_preprocess is None:
            return None

        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)

        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
        return similarity.item()

    def evaluate_tiling(self, img1, img2, direction, width_size=15):
        """
        This evaluation method tests the tiling between 2 images along some axis. It is based on mean absolute gradient
        between the images
        :param img1: source image
        :param img2: target image
        :param direction: 'x' or 'y' - the direction of the tiling
        :param width_size: Width of the tiling
        :return: Tiling score (between 0.0 and 1.0) - best is minimal - 0.0
        """
        score = mean_absolute_gradient(img_1=img1, img_2=img2, direction=direction, width_size=width_size)
        return score

    def evaluate_image_inception(self, img):
        """
        This evaluation method returns the inception score of the image
        :param img1: A generated image
        :return: Inception score
        """
        self._ensure_inception_model()
        image_uint8 = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        transformed_image = self.inception_transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            pred = torch.nn.functional.softmax(self.inception_model(transformed_image), dim=1).cpu().numpy()

        eps = 1e-16
        # Calculate the marginal probability
        p_y = np.mean(pred, axis=0)
        # Calculate the KL divergence
        kl_d = pred * (np.log(pred + eps) - np.log(p_y + eps))
        # Sum over classes
        sum_kl_d = np.sum(kl_d, axis=1)
        # Calculate the mean
        avg_kl_d = np.mean(sum_kl_d)
        # Calculate the inception score
        is_score = np.exp(avg_kl_d)
        return is_score

    def evaluate_image_quality(self, img):
        """
        This function evaluates the quality of an image
        :param img: A generated image
        :return: Quality of an image
        """
        if self.clip_model is None:
            raise RuntimeError("OpenAI CLIP 未成功加载，无法计算图像质量指标。")

        image_uint8 = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        processed_img = preprocess(pil_image).unsqueeze(0)
        processed_img = processed_img.to(self.device)
        # Get CLIP embedding
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_img)
        clip_embedding = image_features.cpu().numpy()

        # Calculate color histogram
        hist = np.histogram(image_uint8.flatten(), bins=256, range=[0, 256])[0]
        hist = hist / np.sum(hist)

        # Calculate mean and median for each channel
        means = np.mean(image_uint8, axis=(0, 1))
        medians = np.median(image_uint8, axis=(0, 1))

        # Calculate Wasserstein distance between histogram and uniform distribution
        uniform_dist = np.ones(256) / 256
        w_dist = wasserstein_distance(hist, uniform_dist)

        # Calculate L1 distance for means and medians from the center value (127.5)
        mean_dist = np.sum(np.abs(means - 127.5))
        median_dist = np.sum(np.abs(medians - 127.5))

        # Calculate the norm of the CLIP embedding
        clip_norm = np.linalg.norm(clip_embedding)

        # Combine the metrics (you may want to adjust the weights)
        cmmd_score = w_dist + 0.1 * mean_dist + 0.1 * median_dist - 0.1 * clip_norm

        return cmmd_score

    def _compute_local_ssim(self, arr1, arr2):
        """使用局部高斯窗口计算单通道 SSIM，并返回整幅图的平均值。"""
        height, width = arr1.shape
        kernel_size = min(11, height, width)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(kernel_size, 3)

        sigma = 1.5 if kernel_size >= 7 else 1.0
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2

        mu1 = cv2.GaussianBlur(arr1, (kernel_size, kernel_size), sigma)
        mu2 = cv2.GaussianBlur(arr2, (kernel_size, kernel_size), sigma)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(arr1 * arr1, (kernel_size, kernel_size), sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(arr2 * arr2, (kernel_size, kernel_size), sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(arr1 * arr2, (kernel_size, kernel_size), sigma) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
        )
        return float(np.mean(ssim_map))

    def evaluate_boundary_ssim(self, img1, img2, direction, strip_width=15):
        """
        计算两个相邻 tile 接缝处边界条带的结构相似度 (SSIM)。
        :param img1: 第一张图 (numpy array, shape HxWxC, 值域 0-1)
        :param img2: 第二张图
        :param direction: 'x' 水平拼接 (img1 右边 — img2 左边)，'y' 垂直拼接
        :param strip_width: 取边界条带的像素宽度
        :return: SSIM 值 (0-1)
        """
        if direction == 'x':
            strip1 = img1[:, -strip_width:, :].astype(np.float64)
            strip2 = img2[:, :strip_width, :].astype(np.float64)
        elif direction == 'y':
            strip1 = img1[-strip_width:, :, :].astype(np.float64)
            strip2 = img2[:strip_width, :, :].astype(np.float64)
        else:
            raise ValueError("direction must be 'x' or 'y'")

        strip1 = np.clip(strip1, 0.0, 1.0)
        strip2 = np.clip(strip2, 0.0, 1.0)

        if strip1.ndim == 2:
            return self._compute_local_ssim(strip1, strip2)

        channel_scores = []
        for channel in range(strip1.shape[2]):
            channel_scores.append(self._compute_local_ssim(strip1[:, :, channel], strip2[:, :, channel]))
        return float(np.mean(channel_scores))

    def evaluate_clip_consistency(self, images):
        """
        计算多个 tile 之间的 CLIP 视觉一致性（两两余弦相似度的平均值）。
        :param images: 图像列表 (list of numpy arrays, 值域 0-1)
        :return: 平均余弦相似度 (0-1)
        """
        if self.clip_model is None or self.clip_preprocess is None:
            return None

        features = []
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            img_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(img_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)

        sims = []
        n = len(features)
        for i in range(n):
            for j in range(i + 1, n):
                sim = torch.nn.functional.cosine_similarity(features[i], features[j]).item()
                sims.append(sim)

        return float(np.mean(sims)) if sims else None
