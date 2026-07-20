import random
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout
from torchvision import transforms

from ..attack import Attack


# ----------------------------------------------------------------------------
# Spatial roll utilities (class-based, replaces standalone shift functions)
# ----------------------------------------------------------------------------
class _AxisRoll:
    """Rolls a 4D tensor by a random step along a chosen spatial axis."""

    def __init__(self, axis: int):
        self._axis = axis

    def __call__(self, tensor):
        extent = tensor.shape[self._axis]
        offset = int(np.random.randint(low=0, high=extent, dtype=np.int32))
        return tensor.roll(offset, dims=self._axis)


def make_vertical_roll():
    return _AxisRoll(axis=2)


def make_horizontal_roll():
    return _AxisRoll(axis=3)


# ----------------------------------------------------------------------------
# Transform registry: maps a keyword to a factory that builds the transform
# ----------------------------------------------------------------------------
class TransformRegistry:
    """A small factory hub that produces augmentation callables on demand."""

    def __init__(self):
        self._builders = {}
        self._register_defaults()

    def register(self, key: str, builder: Callable):
        self._builders[key] = builder

    def _register_defaults(self):
        self.register("rotate", lambda: transforms.RandomRotation(random.randint(0, 180)))
        self.register("blockshuffle", lambda: BlockShuffle(random.randint(2, 5)))
        self.register("resizedpad", lambda: ResizePadResize(round(random.uniform(1.14, 1.66), 2)))
        self.register("random_crop", lambda: transforms.RandomCrop(224, padding=random.randint(0, 30)))
        self.register("vertical_shift", make_vertical_roll)
        self.register("horizontal_shift", make_horizontal_roll)
        self.register("ssm", lambda: SpectrumMixer(round(random.uniform(0.1, 0.9), 1)))
        self.register("ide", lambda: DropoutEnsemble((0, 0.1, 0.2, 0.3, 0.4, 0.5)))

    def build(self, key: str):
        if key not in self._builders:
            raise ValueError(f"Unknown transform: {key}")
        return self._builders[key]()

    def compose(self, keys: List[str]):
        return transforms.Compose([self.build(k) for k in keys])


# ----------------------------------------------------------------------------
# Sampling policy: decides which transform names get combined each round
# ----------------------------------------------------------------------------
@dataclass
class _ShiftExpansion:
    """Turns a generic 'shift' token into concrete axis-roll tokens."""

    _MODES = ("vertical", "horizontal", "both")

    def apply(self, names: List[str]) -> List[str]:
        if "shift" not in names:
            return names
        result = [n for n in names if n != "shift"]
        mode = random.choice(self._MODES)
        if mode == "vertical":
            result.append("vertical_shift")
        elif mode == "horizontal":
            result.append("horizontal_shift")
        else:
            result += ["vertical_shift", "horizontal_shift"]
        return result


class TransformSampler:
    """Draws a random subset of transform names from the active pool."""

    def __init__(self):
        self._shift_expander = _ShiftExpansion()

    @staticmethod
    def pool_for(budget: int) -> List[str]:
        light = ["resizedpad", "random_crop", "shift"]
        heavy = ["rotate", "blockshuffle", "resizedpad",
                 "random_crop", "ssm", "shift", "ide"]
        return light if budget <= 20 else heavy

    def draw(self, pool: List[str]) -> List[str]:
        how_many = random.randint(1, len(pool))
        picked = random.sample(pool, how_many)
        return self._shift_expander.apply(picked)


# ----------------------------------------------------------------------------
# Main attack
# ----------------------------------------------------------------------------
class SIMATT(Attack):

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        num_iter=10,
        transform_num=2000,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="SIMATT",
        **kwargs,
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / num_iter
        self.epoch = num_iter
        self.decay = decay

        self.transform_num = int(transform_num)
        if self.transform_num < 0:
            raise ValueError("transform_num must be non-negative")

        self.using_sampling = self.transform_num > 0
        self._registry = TransformRegistry()
        self._sampler = TransformSampler()
        self._active_pool = self._sampler.pool_for(self.transform_num)

    # -- public helpers kept for backward compatibility -------------------
    def get_huy_basic(self):
        return self._sampler.pool_for(self.transform_num)

    def get_new_SIMATT_string(self):
        return self._sampler.draw(self._active_pool)

    def get_transform(self, transform_name):
        return self._registry.build(transform_name)

    def transform123(self, data, transform_names):
        keys = [transform_names] if isinstance(transform_names, str) else list(transform_names)
        return self._registry.compose(keys)(data)

    # -- gradient machinery ----------------------------------------------
    def _accumulate_sampled_gradients(self, x_near, delta, label):
        """Sums gradients over many randomly-composed augmentations."""
        total = torch.zeros_like(delta)
        self._active_pool = self._sampler.pool_for(self.transform_num)
        for _ in range(self.transform_num):
            keys = self._sampler.draw(self._active_pool)
            perturbed = self.transform123(x_near, keys)
            logits = self.get_logits(perturbed)
            loss = self.get_loss(logits, label)
            total = total + self.get_grad(loss, delta)
        return total

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        base_grad = self.get_surrogate_gradient(data, delta, label)
        if not self.using_sampling:
            return base_grad

        sampled = self._accumulate_sampled_gradients(data + delta, delta, label)
        return (base_grad + sampled) / (self.transform_num + 1)

    def get_surrogate_gradient(self, data, delta, label, **kwargs):
        logits = self.get_logits(data + delta)
        loss = self.get_loss(logits, label)
        return self.get_grad(loss, delta)

    # -- optimization loop, split into stages ----------------------------
    def _single_step(self, data, label, delta, momentum):
        grad = self.get_averaged_gradient(data, delta, label)
        momentum = self.get_momentum(grad, momentum)
        delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta, momentum

    def _run_optimization(self, data, label):
        delta = self.init_delta(data)
        momentum = 0
        for idx in range(self.epoch):
            tic = time.time()
            delta, momentum = self._single_step(data, label, delta, momentum)
            print(f"Iteration {idx + 1}/{self.epoch} cost: {time.time() - tic:.4f} seconds")
        return delta

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        return self._run_optimization(data, label).detach()


# ----------------------------------------------------------------------------
# Augmentation modules
# ----------------------------------------------------------------------------
class BlockShuffle(nn.Module):
    def __init__(self, num_block=4):
        super().__init__()
        self.num_block = num_block

    def _random_partition(self, length):
        weights = np.random.uniform(size=self.num_block)
        sizes = np.round(weights / weights.sum() * length).astype(np.int32)
        sizes[sizes.argmax()] += length - sizes.sum()
        return tuple(sizes)

    def _shuffle_along(self, x, dim):
        parts = list(x.split(self._random_partition(x.size(dim)), dim=dim))
        random.shuffle(parts)
        return parts

    def forward(self, x):
        columns = self._shuffle_along(x, dim=3)
        rebuilt = [torch.cat(self._shuffle_along(col, dim=2), dim=2) for col in columns]
        return torch.cat(rebuilt, dim=3)


class ResizePadResize(nn.Module):
    def __init__(self, resize_rate=1.15):
        super().__init__()
        self.resize_rate = resize_rate

    @staticmethod
    def _rand_int(low, high):
        return int(torch.randint(low=low, high=high, size=(1,), dtype=torch.int32).item())

    def forward(self, x):
        base = x.shape[-1]
        enlarged = int(base * self.resize_rate)

        lo, hi = min(base, enlarged), max(base, enlarged)
        target = self._rand_int(lo, hi)
        stage_one = F.interpolate(x, size=[target, target], mode="bilinear", align_corners=False)

        remainder = enlarged - target
        top = self._rand_int(0, remainder)
        left = self._rand_int(0, remainder)
        margins = [left, remainder - left, top, remainder - top]

        stage_two = F.pad(stage_one, margins, value=0)
        return F.interpolate(stage_two, size=[base, base], mode="bilinear", align_corners=False)


class SpectrumMixer:
    def __init__(self, rho=0.5):
        self.epsilon = 16 / 255
        self.rho = rho

    def dct(self, x, norm=None):
        x_shape = x.shape
        n = x_shape[-1]
        x = x.contiguous().view(-1, n)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        vc = torch.fft.fft(v)

        k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
        w_r = torch.cos(k)
        w_i = torch.sin(k)

        result = vc.real * w_r - vc.imag * w_i
        if norm == "ortho":
            result[:, 0] /= np.sqrt(n) * 2
            result[:, 1:] /= np.sqrt(n / 2) * 2

        return 2 * result.view(*x_shape)

    def idct(self, x, norm=None):
        x_shape = x.shape
        n = x_shape[-1]
        x_v = x.contiguous().view(-1, n) / 2

        if norm == "ortho":
            x_v[:, 0] *= np.sqrt(n) * 2
            x_v[:, 1:] *= np.sqrt(n / 2) * 2

        k = torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * n)
        w_r = torch.cos(k)
        w_i = torch.sin(k)

        v_t_r = x_v
        v_t_i = torch.cat([x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1)

        v_r = v_t_r * w_r - v_t_i * w_i
        v_i = v_t_r * w_i + v_t_i * w_r

        v = torch.cat([v_r.unsqueeze(2), v_i.unsqueeze(2)], dim=2)
        inverse = torch.fft.ifft(torch.complex(real=v[:, :, 0], imag=v[:, :, 1]))

        output = inverse.new_zeros(inverse.shape)
        output[:, ::2] += inverse[:, : n - (n // 2)]
        output[:, 1::2] += inverse.flip([1])[:, : n // 2]
        return output.view(*x_shape).real

    def dct_2d(self, x, norm=None):
        x1 = self.dct(x, norm=norm)
        x2 = self.dct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def idct_2d(self, x, norm=None):
        x1 = self.idct(x, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def __call__(self, x):
        gauss = torch.randn_like(x) * self.epsilon
        x_dct = self.dct_2d(x + gauss)
        mask = torch.rand_like(x) * 2 * self.rho + (1 - self.rho)
        return self.idct_2d(x_dct * mask)


class DropoutEnsemble:
    def __init__(self, dropout_prob=(0, 0.1, 0.2, 0.3, 0.4, 0.5)):
        self.dropout_prob = tuple(dropout_prob)

    def __call__(self, x):
        prob = random.choice(self.dropout_prob)
        return Dropout(p=prob)(x) * (1 - prob)


# ----------------------------------------------------------------------------
# Backward-compatible aliases (so external code importing old names still works)
# ----------------------------------------------------------------------------
blockshuffle = BlockShuffle
resizedpad = ResizePadResize
ssm = SpectrumMixer
ide = DropoutEnsemble
vertical_shift = make_vertical_roll()
horizontal_shift = make_horizontal_roll()
