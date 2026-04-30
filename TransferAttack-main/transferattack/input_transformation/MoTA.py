import random
import time
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TFF
from torchvision import transforms
from torch import nn
from torch.nn import Dropout

from ..utils import *
from ..attack import Attack


# ---------------------------------------------------------------------------
# Image transformation primitives
# ---------------------------------------------------------------------------

def roll_height(x):
    """Randomly roll the input tensor along the height dimension."""
    h = x.shape[2]
    offset = np.random.randint(0, h, dtype=np.int32)
    return x.roll(offset, dims=2)


def roll_width(x):
    """Randomly roll the input tensor along the width dimension."""
    w = x.shape[3]
    offset = np.random.randint(0, w, dtype=np.int32)
    return x.roll(offset, dims=3)


def flip_h(x):
    return x.flip(dims=(2,))


def flip_w(x):
    return x.flip(dims=(3,))


def passthrough(x):
    return x


class Rescale:
    """Divide pixel values by a constant scale factor."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, x):
        return x / self.factor


class Rotation:
    """Rotate the input image by a fixed angle (in degrees)."""

    def __init__(self, deg: float):
        self.deg = deg

    def __call__(self, x):
        return TFF.rotate(img=x, angle=self.deg)


class RandomResizePad:
    """Randomly resize the image, pad to a target size, then restore original resolution.

    This is a variant of the Diverse Input Module (DIM).
    """

    def __init__(self, scale: float = 1.1, prob: float = 0.5):
        self.scale = scale
        self.prob  = prob

    def __call__(self, x):
        orig   = x.shape[-1]
        target = int(orig * self.scale)

        rnd = torch.randint(
            low=min(orig, target),
            high=max(orig, target),
            size=(1,),
            dtype=torch.int32,
        )
        x_rs = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        dh = target - rnd
        dw = target - rnd
        pt = torch.randint(0, dh.item(), (1,), dtype=torch.int32)
        pb = dh - pt
        pl = torch.randint(0, dw.item(), (1,), dtype=torch.int32)
        pr = dw - pl
        x_pad = F.pad(x_rs, [pl.item(), pr.item(), pt.item(), pb.item()], value=0)

        return F.interpolate(x_pad, size=[orig, orig], mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# Auxiliary transform modules
# ---------------------------------------------------------------------------

class BlockShuffle(nn.Module):
    """Partition the image into random-sized blocks and shuffle them spatially.

    Note: the first shuffle pass (along H then W) intentionally discards its
    result to match the behaviour of the original implementation.
    """

    def __init__(self, n_blocks: int = 4):
        super().__init__()
        self.n_blocks = n_blocks

    def _rand_split(self, length: int):
        """Return a tuple of random positive integers that sum to *length*."""
        w     = np.random.uniform(size=self.n_blocks)
        sizes = np.round(w / w.sum() * length).astype(np.int32)
        sizes[sizes.argmax()] += length - sizes.sum()
        return tuple(sizes)

    def _shuffle_dim(self, x, d: int):
        """Split tensor along dimension *d* and return a shuffled list of strips."""
        parts = list(x.split(self._rand_split(x.size(d)), dim=d))
        random.shuffle(parts)
        return parts

    def forward(self, x):
        # First pass along H then W — result discarded (matches original behaviour)
        rows = self._shuffle_dim(x, 2)
        torch.cat(
            [torch.cat(self._shuffle_dim(r, 3), dim=3) for r in rows], dim=2
        )
        # Second pass along W then H — this is the actual output
        cols = self._shuffle_dim(x, 3)
        return torch.cat(
            [torch.cat(self._shuffle_dim(c, 2), dim=2) for c in cols], dim=3
        )


class PadResize(nn.Module):
    """Scale the image up, add random padding, then resize back to the original resolution."""

    def __init__(self, ratio: float = 1.15):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        sz  = x.shape[-1]
        tsz = int(sz * self.ratio)

        rnd = torch.randint(min(sz, tsz), max(sz, tsz), (1,), dtype=torch.int32)
        xr  = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        dh = tsz - rnd
        dw = tsz - rnd
        pt = torch.randint(0, dh.item(), (1,), dtype=torch.int32)
        pb = dh - pt
        pl = torch.randint(0, dw.item(), (1,), dtype=torch.int32)
        pr = dw - pl
        xp = F.pad(xr, [pl.item(), pr.item(), pt.item(), pb.item()], value=0)

        return F.interpolate(xp, size=[sz, sz], mode='bilinear', align_corners=False)


class SpectrumPerturb:
    """Apply random perturbations in the DCT frequency domain (SSM variant).

    The input is transformed to the 2-D DCT domain, multiplied by a random
    spectral mask drawn from U[1-rho, 1+rho], then transformed back via IDCT.

    DCT / IDCT implementation adapted from:
        https://github.com/yuyang-long/SSA/blob/master/dct.py
    """

    def __init__(self, rho: float = 0.5, n_spectrum: int = 10):
        self.eps        = 16 / 255
        self.rho        = rho
        self.n_spectrum = n_spectrum

    def _dct1d(self, x, norm=None):
        """1-D Discrete Cosine Transform (Type II)."""
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v  = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v)

        k   = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0]  /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        return (2 * V).view(*x_shape)

    def _idct1d(self, X, norm=None):
        """1-D Inverse Discrete Cosine Transform (Type III)."""
        x_shape = X.shape
        N       = x_shape[-1]
        X_v     = X.contiguous().view(-1, N) / 2

        if norm == 'ortho':
            X_v[:, 0]  *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k   = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V   = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v   = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2]  += v[:, : N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, : N // 2]

        return x.view(*x_shape).real

    def _dct2d(self, x, norm=None):
        """2-D DCT via two separable 1-D transforms."""
        return self._dct1d(self._dct1d(x, norm).transpose(-1, -2), norm).transpose(-1, -2)

    def _idct2d(self, X, norm=None):
        """2-D IDCT via two separable 1-D inverse transforms."""
        return self._idct1d(self._idct1d(X, norm).transpose(-1, -2), norm).transpose(-1, -2)

    def __call__(self, x):
        gauss  = torch.randn_like(x) * self.eps
        x_freq = self._dct2d(x + gauss)
        gate   = torch.rand_like(x) * 2 * self.rho + (1 - self.rho)
        return self._idct2d(x_freq * gate)


class DropoutEnsemble:
    """Apply a randomly selected dropout rate to the input (IDE variant).

    After dropout, the output is rescaled by (1 - p) to preserve the
    expected activation magnitude.
    """

    def __init__(self, prob_pool=None):
        self.prob_pool = prob_pool or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def __call__(self, x):
        p = random.choice(self.prob_pool)
        return Dropout(p=p)(x) * (1 - p)


# ---------------------------------------------------------------------------
# MoTA Attack
# ---------------------------------------------------------------------------

class MoTA(Attack):
    """Momentum-based Operator Transformation Attack (MoTA).

    MoTA improves adversarial transferability by averaging gradients computed
    over 1000 randomly sampled input transformations per iteration. Each
    gradient is RMS-normalised before accumulation to reduce variance.

    Args:
        model_name:    Name of the surrogate model.
        epsilon:       Maximum perturbation budget (L-inf norm).
        num_iter:      Number of attack iterations.
        decay:         Momentum decay factor.
        targeted:      If True, perform a targeted attack.
        random_start:  If True, initialise delta with uniform noise.
        norm:          Perturbation norm constraint ('linfty' or 'l2').
        loss:          Loss function identifier ('crossentropy', etc.).
        device:        Torch device.
        attack:        Attack name string (used for logging).
    """

    # Active operator name pool (commented-out entries are disabled)
    _OP_POOL = [
        'rotation',
        'block_shuffle',
        'pad_resize',
        'random_crop',
        'spectrum',
        # 'h_shift',
        # 'v_shift',
        # 'dropout',
        # 'blur',
        # 'rescale',
        # 'h_flip',
        # 'v_flip',
    ]

    def __init__(
        self,
        model_name,
        epsilon: float = 16 / 255,
        num_iter: int = 10,
        decay: float = 1.0,
        targeted: bool = False,
        random_start: bool = False,
        norm: str = 'linfty',
        loss: str = 'crossentropy',
        device=None,
        attack: str = 'MoTA',
        # kept for API compatibility with OPS-based configs
        num_sample_neighbor: int = 10,
        num_sample_operator: int = 20,
        **kwargs,
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha                = epsilon / num_iter
        self.epoch                = num_iter
        self.decay                = decay
        self.num_sample_neighbor  = num_sample_neighbor
        self.num_sample_operator  = num_sample_operator

    # ------------------------------------------------------------------
    # Operator factory
    # ------------------------------------------------------------------

    def _sample_op_names(self):
        """Randomly draw a non-empty subset of operator names from the pool."""
        k = random.randint(1, len(self._OP_POOL))
        return random.sample(self._OP_POOL, k)

    def _make_op(self, name: str):
        """Instantiate a single operator by name with randomised hyper-parameters."""
        registry = {
            'rotation':      lambda: Rotation(random.randint(-180, 180)),
            'pad_resize':    lambda: PadResize(round(random.uniform(1.14, 1.66), 2)),
            'spectrum':      lambda: SpectrumPerturb(round(random.uniform(0.1, 0.9), 1)),
            'block_shuffle': lambda: BlockShuffle(random.randint(1, 5)),
            'random_crop':   lambda: transforms.RandomCrop(224, padding=random.randint(0, 30)),
            'h_shift':       lambda: roll_width,
            'v_shift':       lambda: roll_height,
            'blur':          lambda: transforms.GaussianBlur(5, sigma=(0.5, 2.0)),
            'rescale':       lambda: Rescale(random.randint(2, 8)),
            'h_flip':        lambda: flip_w,
            'v_flip':        lambda: flip_h,
            'dropout':       lambda: DropoutEnsemble(),
        }
        if name not in registry:
            raise KeyError(f'Unknown operator: {name}')
        return registry[name]()

    def _apply_ops(self, x, names):
        """Compose a list of operators and apply them sequentially to *x*."""
        if isinstance(names, str):
            names = [names]
        pipeline = transforms.Compose([self._make_op(n) for n in names])
        return pipeline(x)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _base_gradient(self, data, delta, label):
        """Compute the standard gradient on the unperturbed adversarial example."""
        logits = self.get_logits(data + delta)
        loss   = self.get_loss(logits, label)
        return self.get_grad(loss, delta)

    def _compute_avg_gradient(self, data, delta, label):
        """Accumulate RMS-normalised gradients over 1000 randomly transformed inputs.

        The final estimate averages one base gradient and 1000 transformed
        gradients, divided by (num_sample_neighbor * num_sample_operator + 1).
        """
        grad_acc = self._base_gradient(data, delta, label)
        x_adv    = data + delta

        for _ in range(2000):
            op_seq  = self._sample_op_names()
            x_trans = self._apply_ops(x_adv, op_seq)
            logits  = self.get_logits(x_trans)
            loss    = self.get_loss(logits, label)
            g       = self.get_grad(loss, delta)
            rms     = torch.mean(g * g.conj(), dim=(1, 2, 3), keepdim=True).sqrt() + 1e-7
            grad_acc = grad_acc + g / rms

        return grad_acc / (self.num_sample_neighbor * self.num_sample_operator + 1)

    # ------------------------------------------------------------------
    # Main attack loop
    # ------------------------------------------------------------------

    def forward(self, data, label, **kwargs):
        """Run the MoTA attack and return the adversarial perturbation.

        Args:
            data:  (N, C, H, W) input image tensor.
            label: Ground-truth labels (untargeted) or
                   [source_labels, target_labels] (targeted).

        Returns:
            delta (torch.Tensor): Adversarial perturbation, detached from
                the computation graph.
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data  = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta    = self.init_delta(data)
        momentum = 0

        for step in range(self.epoch):
            t_start  = time.time()
            avg_grad = self._compute_avg_gradient(data, delta, label)
            momentum = self.get_momentum(avg_grad, momentum)
            delta    = self.update_delta(delta, data, momentum, self.alpha)
            print(f'[MoTA] step {step + 1}/{self.epoch}  elapsed: {time.time() - t_start:.4f}s')

        return delta.detach()
