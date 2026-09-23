# -*- coding: utf-8 -*-
"""在线合成退化的 RGB 修复数据；可保留源图验证或显式使用全部训练源图。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from data.mim_dataset import list_images, read_manifest


PROFILES = ("identity", "blur", "illumination", "mixed")


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_rgb(image: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """复用主线 letterbox；返回 [0,1] RGB 和排除反射填充的有效区。"""
    boxed, _, pad_h, pad_w = letterbox(image, image_size)
    valid = np.zeros((image_size, image_size), dtype=np.float32)
    valid[:image_size - pad_h, :image_size - pad_w] = 1
    return boxed.astype(np.float32) / 255.0, valid


def degrade_rgb(clean: np.ndarray, rng: np.random.Generator, profile: str, cfg: dict) -> np.ndarray:
    """仅改变外观；参考图取自用户确认基本清晰的训练集。"""
    if profile not in PROFILES:
        raise ValueError(f"unknown degradation profile: {profile}")
    image = clean.copy()
    if profile in {"blur", "mixed"}:
        sigma = float(rng.uniform(*cfg["blur_sigma"]))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        if rng.random() < cfg["resample_probability"]:
            scale = float(rng.uniform(*cfg["resample_scale"]))
            h, w = image.shape[:2]
            small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if profile in {"illumination", "mixed"}:
        gamma = float(rng.uniform(*cfg["gamma"]))
        exposure = float(rng.uniform(*cfg["exposure"]))
        gains = rng.uniform(*cfg["channel_gain"], size=(1, 1, 3)).astype(np.float32)
        # 缓慢变化的照明场；不使用测试集的明度统计作恢复目标。
        amount = float(cfg["illumination_strength"])
        field = rng.uniform(1 - amount, 1 + amount, size=(2, 2)).astype(np.float32)
        field = cv2.resize(field, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        image = np.power(np.clip(image, 0, 1), gamma) * exposure * gains * field[..., None]
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32)


def local_reconstruction_mask(image: np.ndarray, valid: np.ndarray,
                              rng: np.random.Generator, cfg: dict,
                              strength: float) -> tuple[np.ndarray, np.ndarray]:
    """以输入自身的低频内容柔和替换局部区域；不读取清晰目标或晶界标签。

    随机位置同时覆盖边界与内部。返回实际混合权重，零权重位置逐值保持输入。
    低频替换保留亮度线索，避免黑块与有界残差输出之间的冲突。
    """
    h, w = valid.shape
    mask = np.zeros((h, w), dtype=np.float32)
    count = float(valid.sum())
    if count == 0 or strength == 0:
        return image.copy(), mask
    target = float(rng.uniform(*cfg["coverage"])) * count
    covered = 0.0
    for _ in range(512):
        if covered >= target:
            break
        size = int(rng.integers(cfg["patch_size"][0], cfg["patch_size"][1] + 1))
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        y1, y2 = max(0, cy - size // 2), min(h, cy + (size + 1) // 2)
        x1, x2 = max(0, cx - size // 2), min(w, cx + (size + 1) // 2)
        yy = (np.arange(y1, y2, dtype=np.float32) - cy) / (size / 2)
        xx = (np.arange(x1, x2, dtype=np.float32) - cx) / (size / 2)
        radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        patch = np.clip((1 - radius) / 0.25, 0, 1) * valid[y1:y2, x1:x2]
        previous = mask[y1:y2, x1:x2]
        combined = np.maximum(previous, patch)
        covered += float((combined - previous).sum())
        mask[y1:y2, x1:x2] = combined
    sigma = float(rng.uniform(*cfg["lowpass_sigma"]))
    lowpass = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    mask *= strength * float(rng.uniform(*cfg["blend"]))
    restored_input = image + mask[..., None] * (lowpass - image)
    return np.ascontiguousarray(np.clip(restored_input, 0, 1), dtype=np.float32), mask


def _add_achromatic_noise(image: np.ndarray, valid: np.ndarray,
                         rng: np.random.Generator, sigma: float) -> np.ndarray:
    """三通道共用噪声，只改变有效内容；保留原来的清晰参考和填充像素。"""
    noise = rng.standard_normal(valid.shape).astype(np.float32) * sigma
    noisy = np.clip(image + noise[..., None], 0, 1)
    return np.ascontiguousarray(np.where(valid[..., None] > 0, noisy, image), dtype=np.float32)


class RGBRestorationDataset(Dataset):
    """训练为模型尺度上的随机裁块；验证为固定退化的完整 letterbox 图。

    随机数仅依赖 seed/epoch/源图索引，多 worker 和续训不会重复固定的训练退化。
    验证逐源图输出四种情况，单列清晰图漂移，避免只挑合成模糊图报告。
    """

    def __init__(self, data_dir: str, holdout_manifest: str | None = None, *, split: str,
                 image_size: int, crop_size: int, degradation: dict, seed: int = 42,
                 masked_pretraining: dict | None = None,
                 training_noise: dict | None = None, split_policy: str = "holdout"):
        if split not in {"train", "holdout"}:
            raise ValueError("split must be train or holdout")
        if split_policy not in {"holdout", "all_train"}:
            raise ValueError("split_policy must be holdout or all_train")
        if split_policy == "all_train" and split != "train":
            raise ValueError("all_train does not define a holdout split")
        if not 4 <= crop_size <= image_size:
            raise ValueError("expected 4 <= crop_size <= image_size")
        paths = list_images(data_dir)
        if split_policy == "all_train":
            # 全量模式不读取清单，旧保留图和其他允许的训练源图同样进入训练。
            self.samples = paths
        else:
            if not holdout_manifest:
                raise ValueError("holdout split_policy requires holdout_manifest")
            holdout = read_manifest(holdout_manifest)
            stems = {Path(path).stem for path in paths}
            if not holdout or holdout - stems:
                raise ValueError(f"empty/incomplete holdout manifest: {sorted(holdout - stems)}")
            self.samples = [path for path in paths if (Path(path).stem in holdout) == (split == "holdout")]
        if not self.samples:
            raise ValueError(f"no {split} source images in {data_dir}")
        self.split, self.image_size, self.crop_size = split, int(image_size), int(crop_size)
        self.split_policy = split_policy
        self.degradation, self.seed, self.epoch = degradation, int(seed), 0
        self.masked_pretraining = masked_pretraining
        self.training_noise = training_noise
        if masked_pretraining:
            for name, lower, upper in (("coverage", 0, 0.5), ("blend", 0, 1),
                                       ("patch_size", 2, image_size), ("lowpass_sigma", 0, image_size)):
                values = masked_pretraining[name]
                if len(values) != 2 or not lower < values[0] <= values[1] <= upper:
                    raise ValueError(f"invalid masked_pretraining.{name}: {values}")
            if (not 0 < masked_pretraining["start_strength"] <= 1
                    or int(masked_pretraining["epochs"]) != masked_pretraining["epochs"]
                    or masked_pretraining["epochs"] < 1):
                raise ValueError("masked pretraining requires positive epochs and 0 < start_strength <= 1")
            probability = masked_pretraining.get("continuation_probability", 0.0)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("masked_pretraining.continuation_probability must be in [0,1]")
        if training_noise is not None:
            probability = training_noise["probability"]
            sigmas = np.asarray(training_noise["sigma"], dtype=np.float64)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("training_noise.probability must be in [0,1]")
            if (sigmas.shape != (2,) or not np.all(np.isfinite(sigmas))
                    or not 0 <= sigmas[0] <= sigmas[1] <= 1):
                raise ValueError("training_noise.sigma requires finite 0 <= low <= high <= 1")
        self.probabilities = np.asarray(degradation["profile_probabilities"], dtype=np.float64)
        if (self.probabilities.shape != (4,) or np.any(self.probabilities < 0)
                or not np.isclose(self.probabilities.sum(), 1)):
            raise ValueError("four profile probabilities must be nonnegative and sum to one")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def masking_strength(self) -> float:
        if self.split != "train" or not self.masked_pretraining:
            return 0.0
        cfg = self.masked_pretraining
        if self.epoch >= cfg["epochs"]:
            return 1.0 if self.masking_probability() > 0 else 0.0
        progress = self.epoch / max(1, cfg["epochs"] - 1)
        return float(cfg["start_strength"] + (1 - cfg["start_strength"]) * progress)

    def masking_probability(self) -> float:
        """给定 blur 样本的局部任务概率；不改变 identity/blur 的原采样比例。"""
        if self.split != "train" or not self.masked_pretraining:
            return 0.0
        if self.epoch < self.masked_pretraining["epochs"]:
            return 1.0
        return float(self.masked_pretraining.get("continuation_probability", 0.0))

    def __len__(self) -> int:
        return len(self.samples) * (len(PROFILES) if self.split == "holdout" else 1)

    def __getitem__(self, index: int) -> dict:
        training = self.split == "train"
        source_index = index if training else index // len(PROFILES)
        rng = np.random.default_rng(np.random.SeedSequence([
            self.seed, self.epoch if training else 0, index, 0 if training else 1,
        ]))
        profile = str(rng.choice(PROFILES, p=self.probabilities)) if training else PROFILES[index % 4]
        clean, valid = prepare_rgb(read_rgb(self.samples[source_index]), self.image_size)
        vh, vw = int(valid[:, 0].sum()), int(valid[0].sum())
        # 在真实内容上退化，再补齐镜像，避免给填充区学出虚构监督。
        degraded = degrade_rgb(clean[:vh, :vw], rng, profile, self.degradation)
        degraded = cv2.copyMakeBorder(degraded, 0, self.image_size - vh, 0, self.image_size - vw,
                                      cv2.BORDER_REFLECT)
        if training:
            size = self.crop_size
            y = int(rng.integers(0, max(0, vh - size) + 1))
            x = int(rng.integers(0, max(0, vw - size) + 1))
            clean, degraded, valid = [a[y:y + size, x:x + size] for a in (clean, degraded, valid)]
            turns, flip = int(rng.integers(4)), bool(rng.integers(2))
            arrays = [np.rot90(a, turns) for a in (clean, degraded, valid)]
            clean, degraded, valid = [a[:, ::-1] if flip else a for a in arrays]
        mask_mean = 0.0
        strength = self.masking_strength()
        apply_mask = bool(strength and profile == "blur")
        probability = self.masking_probability()
        if apply_mask and probability < 1:
            # 只决定后续局部任务是否出现，沿用原随机流 2 生成具体损伤。
            choice_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 4]))
            apply_mask = bool(choice_rng.random() < probability)
        if apply_mask:
            # 独立随机流：不改变 v3 的退化、裁块、翻转或采样次序。
            mask_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 2]))
            degraded, mask = local_reconstruction_mask(degraded, valid, mask_rng,
                                                        self.masked_pretraining, strength)
            mask_mean = float(mask.sum() / max(1, valid.sum()))
        noise_sigma = 0.0
        if training and self.training_noise is not None and profile == "blur":
            # 在裁块/翻转/局部任务之后添加；独立随机流不移动配对内容。
            noise_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 3]))
            if noise_rng.random() < self.training_noise["probability"]:
                noise_sigma = float(noise_rng.uniform(*self.training_noise["sigma"]))
                if noise_sigma:
                    degraded = _add_achromatic_noise(degraded, valid, noise_rng, noise_sigma)
        result = {
            "input": torch.from_numpy(np.ascontiguousarray(degraded.transpose(2, 0, 1))),
            "target": torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1))),
            "valid": torch.from_numpy(np.ascontiguousarray(valid[None])),
            "profile": profile,
            "source": Path(self.samples[source_index]).name,
        }
        if training and self.masked_pretraining:
            result["mask_mean"] = mask_mean
        if training and self.training_noise is not None:
            result["noise_sigma"] = noise_sigma
        return result


class LocalDamageDiagnosticDataset(Dataset):
    """固定保留图中心裁块上的局部损伤探针；只用于评估，不参与训练或选优。"""

    def __init__(self, holdout: RGBRestorationDataset, cfg: dict):
        if holdout.split != "holdout":
            raise ValueError("local damage diagnostics require held-out sources")
        if not 0 < cfg["strength"] <= 1 or not 0 <= cfg["affected_threshold"] < 1:
            raise ValueError("invalid local damage diagnostic strength/threshold")
        self.holdout, self.cfg = holdout, cfg

    def __len__(self):
        return len(self.holdout.samples)

    def __getitem__(self, index):
        sample_index = index * len(PROFILES) + PROFILES.index("blur")
        sample = self.holdout[sample_index]
        valid = sample["valid"]
        vh, vw = int(valid[0, :, 0].sum()), int(valid[0, 0, :].sum())
        size = self.holdout.crop_size
        y, x = max(0, (vh - size) // 2), max(0, (vw - size) // 2)
        cropped = {key: sample[key][:, y:y + size, x:x + size].clone()
                   for key in ("input", "target", "valid")}
        normal = cropped["input"].permute(1, 2, 0).numpy()
        region = cropped["valid"][0].numpy()
        # 固定为旧版附加诊断的随机流，不随当前训练 epoch 改变。
        rng = np.random.default_rng(np.random.SeedSequence([
            self.holdout.seed, self.cfg["seed_epoch"], sample_index, 2,
        ]))
        damaged, alpha = local_reconstruction_mask(normal, region, rng,
                                                     self.cfg["mask"], self.cfg["strength"])
        affected = (alpha > self.cfg["affected_threshold"]).astype(np.float32) * region
        if affected.sum() == 0:
            raise ValueError(f"empty local damage region: {sample['source']}")
        cropped.update(input=torch.from_numpy(damaged.transpose(2, 0, 1).copy()),
                       affected=torch.from_numpy(affected[None]), source=sample["source"])
        return cropped


class NoiseDiagnosticDataset(Dataset):
    """保留图的固定模糊中心裁块和噪声对照，只诊断抗噪性，不参与选优。"""

    def __init__(self, holdout: RGBRestorationDataset, cfg: dict):
        if holdout.split != "holdout":
            raise ValueError("noise diagnostics require held-out sources")
        sigmas = np.asarray(cfg["sigmas"], dtype=np.float64)
        if (sigmas.ndim != 1 or len(sigmas) == 0 or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0) or np.any(sigmas > 1)):
            raise ValueError("noise diagnostic sigmas must be nonempty finite values in (0,1]")
        self.holdout, self.sigmas = holdout, tuple(float(value) for value in sigmas)

    def __len__(self):
        return len(self.holdout.samples) * len(self.sigmas)

    def __getitem__(self, index):
        source_index, sigma_index = divmod(index, len(self.sigmas))
        sample = self.holdout[source_index * len(PROFILES) + PROFILES.index("blur")]
        valid = sample["valid"]
        vh, vw = int(valid[0, :, 0].sum()), int(valid[0, 0, :].sum())
        size = self.holdout.crop_size
        y, x = max(0, (vh - size) // 2), max(0, (vw - size) // 2)
        cropped = {key: sample[key][:, y:y + size, x:x + size].clone()
                   for key in ("input", "target", "valid")}
        normal = cropped["input"].permute(1, 2, 0).numpy()
        region = cropped["valid"][0].numpy()
        # 同一源图的各 sigma 共享基础噪声；不依赖训练轮次或全局 RNG。
        rng = np.random.default_rng(np.random.SeedSequence([self.holdout.seed, source_index, 5]))
        sigma = self.sigmas[sigma_index]
        noisy = _add_achromatic_noise(normal, region, rng, sigma)
        cropped.update(normal_input=cropped["input"],
                       input=torch.from_numpy(noisy.transpose(2, 0, 1).copy()),
                       source=sample["source"], noise_sigma=sigma)
        return cropped
