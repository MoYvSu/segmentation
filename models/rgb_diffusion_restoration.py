# -*- coding: utf-8 -*-
"""像素域条件残差扩散：预测清晰 RGB，保持带噪状态及桥接运算为 FP32。

借鉴 ResShift 的残差桥，不加载其网络、VQGAN 或感知模型权重。
q(x_t | x_0, y) = N((1-eta_t)x_0 + eta_t*y, kappa^2*eta_t I)。
本项目明确设 eta_0=0、eta_T=1，使部署起点 y+kappa*noise 与训练一致。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.rgb_restoration import NAFBlock


CHECKPOINT_FORMAT_DIFFUSION = "rgb_restoration_diffusion_v1"


class TimeConditionedNAFStage(nn.Module):
    """每个尺度先按时间调制特征，再用与 v6 相同的 NAF 残差块处理。"""

    def __init__(self, channels: int, block_count: int, time_dim: int):
        super().__init__()
        self.time_affine = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))
        self.blocks = nn.Sequential(*(NAFBlock(channels) for _ in range(block_count)))

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_affine(embedding).chunk(2, dim=1)
        features = features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.blocks(features)


class PixelDiffusionRGBRestorer(nn.Module):
    """六通道状态/条件网络，外部仍为三通道 RGB 修复接口。

    t_index 使用 1..num_steps。denoise 输出未截断的 x0 估计；仅完整
    采样结束后 clamp 到 [0, 1]。strength=0 返回原对象且不消耗随机数。
    网络可由调用方放在 autocast 中，桥接系数与采样状态始终为 FP32。
    """

    checkpoint_format = CHECKPOINT_FORMAT_DIFFUSION

    def __init__(
        self,
        width: int = 32,
        encoder_blocks: tuple[int, ...] | list[int] = (1, 2, 4),
        middle_blocks: int = 4,
        decoder_blocks: tuple[int, ...] | list[int] = (2, 2, 1),
        num_steps: int = 16,
        kappa: float = 0.1,
        eta_start: float = 0.001,
        schedule_power: float = 0.3,
        time_dim: int = 128,
    ):
        super().__init__()
        if not isinstance(width, int) or width < 4:
            raise ValueError("width must be an integer >= 4")
        encoder_blocks, decoder_blocks = tuple(encoder_blocks), tuple(decoder_blocks)
        if not encoder_blocks or len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("encoder_blocks and decoder_blocks must have the same nonzero length")
        if any(not isinstance(count, int) or count < 1
               for count in (*encoder_blocks, middle_blocks, *decoder_blocks)):
            raise ValueError("all block counts must be positive integers")
        if not isinstance(num_steps, int) or num_steps < 2:
            raise ValueError("num_steps must be an integer >= 2")
        if (not math.isfinite(kappa) or kappa <= 0 or not 0 < eta_start < 1
                or not math.isfinite(schedule_power) or schedule_power <= 0):
            raise ValueError("kappa and schedule_power must be finite positive; eta_start must be in (0, 1)")
        if not isinstance(time_dim, int) or time_dim < 4 or time_dim % 2:
            raise ValueError("time_dim must be an even integer >= 4")
        self.model_config = {
            "architecture": "pixel_diffusion_nafnet", "width": width,
            "encoder_blocks": list(encoder_blocks), "middle_blocks": middle_blocks,
            "decoder_blocks": list(decoder_blocks), "num_steps": num_steps,
            "kappa": float(kappa), "eta_start": float(eta_start),
            "schedule_power": float(schedule_power), "time_dim": time_dim,
        }
        self.num_steps, self.kappa, self.time_dim = num_steps, float(kappa), time_dim
        self.padding_multiple = 2 ** len(encoder_blocks)

        # sqrt_eta(t)=sqrt(eta_start)*exp(log(1/sqrt(eta_start))*u**power),
        # u=(t-1)/(T-1)。显式保留 t=0；最后端点精确设为 1。
        progress = torch.linspace(0, 1, num_steps, dtype=torch.float64)
        sqrt_eta = math.sqrt(eta_start) * torch.exp(
            math.log(1 / math.sqrt(eta_start)) * progress.pow(schedule_power))
        eta = torch.cat((torch.zeros(1, dtype=torch.float64), sqrt_eta.square())).float()
        eta[1], eta[-1] = float(eta_start), 1.0
        if not torch.all(eta[1:] > eta[:-1]):
            raise ValueError("schedule must be strictly increasing in FP32")
        self.register_buffer("eta", eta)

        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv2d(6, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(TimeConditionedNAFStage(channels, count, time_dim))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle = TimeConditionedNAFStage(channels, middle_blocks, time_dim)
        for count in decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(TimeConditionedNAFStage(channels, count, time_dim))
        self.correction = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    @staticmethod
    def _check_rgb(tensor: torch.Tensor) -> None:
        if tensor.ndim != 4 or tensor.shape[0] < 1 or tensor.shape[1] != 3 or min(tensor.shape[-2:]) < 1:
            raise ValueError("expected nonempty B x 3 x H x W RGB tensor")
        if not tensor.is_floating_point():
            raise ValueError("RGB tensor must have floating dtype")

    def _check_inputs(self, tensor: torch.Tensor, condition: torch.Tensor, t_index: torch.Tensor) -> None:
        self._check_rgb(tensor)
        self._check_rgb(condition)
        if tensor.shape != condition.shape or tensor.device != condition.device:
            raise ValueError("state and condition must have matching shape and device")
        if (t_index.shape != (tensor.shape[0],) or t_index.dtype != torch.long
                or t_index.device != tensor.device):
            raise ValueError("t_index must be B long indices on the image device")
        if bool((t_index < 1).any()) or bool((t_index > self.num_steps).any()):
            raise ValueError("t_index must be in [1, num_steps]")

    def _time_embedding(self, t_index: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(
            half, device=t_index.device, dtype=torch.float32) / (half - 1))
        phase = (t_index.float() * (1000.0 / self.num_steps))[:, None] * frequency[None]
        return self.time_mlp(torch.cat((phase.sin(), phase.cos()), dim=1))

    def denoise(self, xt: torch.Tensor, condition: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        """预测清晰 x0；不把带噪状态或预测裁切到 RGB 区间。"""
        self._check_inputs(xt, condition, t_index)
        h, w = xt.shape[-2:]
        ph, pw = (-h) % self.padding_multiple, (-w) % self.padding_multiple
        features = torch.cat((xt.float(), condition.float()), dim=1)
        if ph or pw:
            mode = "reflect" if h > ph and w > pw else "replicate"
            features = F.pad(features, (0, pw, 0, ph), mode=mode)
        embedding = self._time_embedding(t_index)
        features, skips = self.stem(features), []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features, embedding)
            skips.append(features)
            features = down(features)
        features = self.middle(features, embedding)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip, embedding)
        return condition.float() + self.correction(features)[..., :h, :w].float()

    def q_sample(
        self, x0: torch.Tensor, condition: torch.Tensor, t_index: torch.Tensor,
        noise: torch.Tensor | None = None, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """从给定清晰图和退化图构造随机训练步骤；可传固定 noise 重放。"""
        self._check_inputs(x0, condition, t_index)
        if noise is None:
            noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
        elif noise.shape != x0.shape or noise.device != x0.device:
            raise ValueError("noise must match x0 shape and device")
        eta = self.eta[t_index].float()[:, None, None, None]
        return (1 - eta) * x0.float() + eta * condition.float() + self.kappa * eta.sqrt() * noise.float()

    def posterior_mean_variance(
        self, xt: torch.Tensor, x0_pred: torch.Tensor, t_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """给定 x0 估计的精确桥后验；t=1 的方差为零、均值等于 x0。"""
        self._check_inputs(xt, x0_pred, t_index)
        eta = self.eta[t_index].float()[:, None, None, None]
        eta_previous = self.eta[t_index - 1].float()[:, None, None, None]
        ratio = eta_previous / eta
        mean = ratio * xt.float() + (1 - ratio) * x0_pred.float()
        variance = self.kappa ** 2 * eta_previous * (eta - eta_previous) / eta
        return mean, variance

    def forward(
        self, y: torch.Tensor, strength: float = 1.0, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """完整 T 步随机后验采样；部署/缩略图应在 no_grad 或 inference_mode 中。"""
        self._check_rgb(y)
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        if strength == 0:
            return y
        condition = y.float()
        state = condition + self.kappa * torch.randn(
            y.shape, device=y.device, dtype=torch.float32, generator=generator)
        for step in range(self.num_steps, 0, -1):
            t_index = torch.full((y.shape[0],), step, device=y.device, dtype=torch.long)
            x0_pred = self.denoise(state, condition, t_index)
            state, variance = self.posterior_mean_variance(state, x0_pred, t_index)
            if step > 1:
                state = state + variance.sqrt() * torch.randn(
                    y.shape, device=y.device, dtype=torch.float32, generator=generator)
        return (condition + strength * (state - condition)).clamp(0, 1)
