import collections
from itertools import repeat
import numpy as np
import torch
import torch.jit
from torch import conv2d, conv_transpose2d
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from tqdm import trange
from nodes import InpaintModelConditioning
import comfy.utils
from comfy.utils import ProgressBar
from comfy.model_management import get_torch_device
import folder_paths

class EasyDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as ke:
            raise AttributeError(name) from ke

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

activation_funcs = {
    "linear": EasyDict(
        func=lambda x, **_: x,
        def_alpha=0,
        def_gain=1,
        cuda_idx=1,
        ref="",
        has_2nd_grad=False,
    ),
    "relu": EasyDict(
        func=lambda x, **_: torch.nn.functional.relu(x),
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=2,
        ref="y",
        has_2nd_grad=False,
    ),
    "lrelu": EasyDict(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2,
        def_gain=np.sqrt(2),
        cuda_idx=3,
        ref="y",
        has_2nd_grad=False,
    ),
    "tanh": EasyDict(
        func=lambda x, **_: torch.tanh(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=4,
        ref="y",
        has_2nd_grad=True,
    ),
    "sigmoid": EasyDict(
        func=lambda x, **_: torch.sigmoid(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=5,
        ref="y",
        has_2nd_grad=True,
    ),
    "elu": EasyDict(
        func=lambda x, **_: torch.nn.functional.elu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=6,
        ref="y",
        has_2nd_grad=True,
    ),
    "selu": EasyDict(
        func=lambda x, **_: torch.nn.functional.selu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=7,
        ref="y",
        has_2nd_grad=True,
    ),
    "softplus": EasyDict(
        func=lambda x, **_: torch.nn.functional.softplus(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=8,
        ref="y",
        has_2nd_grad=True,
    ),
    "swish": EasyDict(
        func=lambda x, **_: torch.sigmoid(x) * x,
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=9,
        ref="x",
        has_2nd_grad=True,
    ),
}

def _bias_act_ref(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None):
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0

    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)]).to(x.device)

    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    gain = float(gain)
    if gain != 1:
        x = x * gain

    if clamp >= 0:
        x = x.clamp(-clamp, clamp)

    return x

def bias_act(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None, impl="ref"):
    assert isinstance(x, torch.Tensor)
    assert impl in ["ref", "cuda"]

    return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)

class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation="linear",
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias

        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = x.matmul(w.t().to(x.device))
            out = x + b.reshape(
                [-1 if i == x.ndim - 1 else 1 for i in range(x.ndim)]
            ).to(x.device)
        else:
            x = x.matmul(w.t().to(x.device))
            out = bias_act(x, b, act=self.activation, dim=x.ndim - 1).to(x.device)

        return out

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class MappingNet(nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=8,
        embed_features=None,
        layer_features=None,
        activation="lrelu",
        lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )

        return x

def setup_filter(
    f,
    device=torch.device("cpu"),
    normalize=True,
    flip_filter=False,
    gain=1,
    separable=None,
):
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)

    return f

def _get_filter_size(f):
    if f is None:
        return 1, 1

    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]

    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh

def _get_weight_shape(w):
    shape = [int(sz) for sz in w.shape]
    return shape

def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape

    upx, upy = up, up
    downx, downy = down, down

    padx0, padx1, pady0, pady1 = padding[0], padding[1], padding[2], padding[3]

    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    x = torch.nn.functional.pad(
        x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
    )
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    x = x[:, :, ::downy, ::downx]
    return x

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl="cuda"):
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)

    if not flip_weight:
        w = w.flip([2, 3])

    if (
        kw == 1
        and kh == 1
        and stride == 1
        and padding in [0, [0, 0], (0, 0)]
        and not transpose
    ):
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape(
                    [in_shape[0], in_channels_per_group, -1]
                )
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = conv2d(x, w, groups=groups)
            return x.to(memory_format=torch.channels_last)

    op = conv_transpose2d if transpose else conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)

def conv2d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype == x.dtype)
    assert f is None or (
        isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32
    )
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)

    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = _get_filter_size(f)

    px0, px1, py0, py1 = padding, padding, padding, padding

    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d(
            x=x, f=f, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter
        )
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d(
            x=x,
            f=f,
            up=up,
            padding=[px0, px1, py0, py1],
            gain=up**2,
            flip_filter=flip_filter,
        )
        return x

    if down > 1 and up == 1:
        x = upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(
            x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight
        )
        return x

    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(
                groups * in_channels_per_group, out_channels // groups, kh, kw
            )
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=[pyt, pxt],
            groups=groups,
            transpose=True,
            flip_weight=(not flip_weight),
        )
        x = upfirdn2d(
            x=x,
            f=f,
            padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
            gain=up**2,
            flip_filter=flip_filter,
        )
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(
                x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight
            )

    x = upfirdn2d(
        x=x,
        f=(f if up > 1 else None),
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter,
    )
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x

class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation="linear",
        up=1,
        down=1,
        resample_filter=[
            1,
            3,
            3,
            1,
        ],
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )
        return out

class Conv2dLayerPartial(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation="linear",
        up=1,
        down=1,
        resample_filter=[
            1,
            3,
            3,
            1,
        ],
        conv_clamp=None,
        trainable=True,
    ):
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            activation,
            up,
            down,
            resample_filter,
            conv_clamp,
            trainable,
        )

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size**2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None

def token2feature(x, x_size):
    B, _, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x

def feature2token(x):
    B, C, _, _ = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            down=down,
        )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            up=up,
        )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        down_ratio=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)
            ).masked_fill(attn_mask_windows == 1, 0.0)
            with torch.no_grad():
                mask_windows = torch.clamp(
                    torch.sum(mask_windows, dim=1, keepdim=True), 0, 1
                ).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(
            in_features=in_features, out_features=hidden_features, activation="lrelu"
        )
        self.fc2 = FullyConnectedLayer(
            in_features=hidden_features, out_features=out_features
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        down_ratio=1,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.fuse = FullyConnectedLayer(
            in_features=dim * 2, out_features=dim, activation="lrelu"
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )

        return attn_mask

    def forward(self, x, x_size, mask=None):
        H, W = x_size
        B, _, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                shifted_mask = torch.roll(
                    mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        x_windows = window_partition(
            shifted_x, self.window_size
        )
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.attn_mask
            )
        else:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                mask = torch.roll(
                    shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        down_ratio=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    down_ratio=down_ratio,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation="lrelu"
        )

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        demodulate=True,
        up=1,
        down=1,
        resample_filter=[
            1,
            3,
            3,
            1,
        ],
        conv_clamp=None,
    ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1).to(x.device)
        weight = self.weight.to(x.device) * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch,
        )
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out

class StyleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        resolution,
        kernel_size=3,
        up=1,
        use_noise=False,
        activation="lrelu",
        resample_filter=[
            1,
            3,
            3,
            1,
        ],
        conv_clamp=None,
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            up=up,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(self, x, style, noise_mode="random", gain=1):
        x = self.conv(x, style)

        assert noise_mode in ["random", "const", "none"]

        if self.use_noise:
            if noise_mode == "random":
                xh, xw = x.size()[-2:]
                noise = (
                    torch.randn([x.shape[0], 1, xh, xw], device=x.device)
                    * self.noise_strength
                )
            if noise_mode == "const":
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )

        return out

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl="cuda"):
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(
        x,
        f,
        up=up,
        padding=p,
        flip_filter=flip_filter,
        gain=gain * upx * upy,
        impl=impl,
    )

class ToRGB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        kernel_size=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        demodulate=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upsample2d(skip, self.resample_filter)
            out = out + skip

        return out

class DecStyleBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, style, skip, noise_mode="random"):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img

class FirstStage(nn.Module):
    def __init__(
        self,
        img_channels,
        img_resolution=256,
        dim=180,
        w_dim=512,
        use_noise=False,
        demodulate=True,
        activation="lrelu",
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels + 1,
            out_channels=dim,
            kernel_size=3,
            activation=activation,
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        
        for i in range(down_time):
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )

        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=[res, res],
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=merge,
                )
            )

        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(
            in_features=dim, out_features=dim * 2, activation=activation
        )
        self.ws_style = FullyConnectedLayer(
            in_features=w_dim, out_features=dim, activation=activation
        )
        self.to_square = FullyConnectedLayer(
            in_features=dim, out_features=16 * 16, activation=activation
        )

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):
            res = res * 2
            self.dec_conv.append(
                DecStyleBlock(
                    res,
                    dim,
                    dim,
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                )
            )

    def forward(self, images_in, masks_in, ws, noise_mode="random"):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)
        skips.append(x)
        for i, block in enumerate(self.enc_conv):
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True).to(x.device)
                ws = self.ws_style(ws[:, -1]).to(x.device)
                add_n = self.to_square(ws).unsqueeze(1).to(x.device)
                add_n = (
                    F.interpolate(
                        add_n, size=x.size(1), mode="linear", align_corners=False
                    )
                    .squeeze(1)
                    .unsqueeze(-1)
                ).to(x.device)
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                ).to(x.device)
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv):
            x, img = block(
                x, img, style, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode
            )

        img = img * (1 - masks_in) + images_in * masks_in

        return img

class EncFromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

class ConvBlockDown(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):
        super().__init__()

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2**stage]

class Encoder(nn.Module):
    def __init__(
        self,
        res_log2,
        img_channels,
        activation,
        patch_size=5,
        channels=16,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.resolution = []

        for i in range(res_log2, 3, -1):
            res = 2**i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, "EncConv_Block_%dx%d" % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, "EncConv_Block_%dx%d" % (res, res))(x)
            out[res_log2] = x

        return out

class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(
            in_features=in_channels, out_features=out_channels, activation=activation
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x

def get_style_code(a, b):
    return torch.cat([a, b.to(a.device)], dim=1)

class DecBlockFirstV2(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img

class DecBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, ws, gs, E_features, noise_mode="random"):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img

class Decoder(nn.Module):
    def __init__(
        self, res_log2, activation, style_dim, use_noise, demodulate, img_channels
    ):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(
            4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels
        )
        for res in range(5, res_log2 + 1):
            setattr(
                self,
                "Dec_%dx%d" % (2**res, 2**res),
                DecBlock(
                    res,
                    nf(res - 1),
                    nf(res),
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                ),
            )
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, "Dec_%dx%d" % (2**res, 2**res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img

class SynthesisNet(nn.Module):
    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels=3,
        channel_base=32768,
        channel_decay=1.0,
        channel_max=512,
        activation="lrelu",
        drop_rate=0.5,
        use_noise=False,
        demodulate=True,
    ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        self.first_stage = FirstStage(
            img_channels,
            img_resolution=img_resolution,
            w_dim=w_dim,
            use_noise=False,
            demodulate=demodulate,
        )

        self.enc = Encoder(
            resolution_log2, img_channels, activation, patch_size=5, channels=16
        )
        self.to_square = FullyConnectedLayer(
            in_features=w_dim, out_features=16 * 16, activation=activation
        )
        self.to_style = ToStyle(
            in_channels=nf(4),
            out_channels=nf(2) * 2,
            activation=activation,
            drop_rate=drop_rate,
        )
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(
            resolution_log2, activation, style_dim, use_noise, demodulate, img_channels
        )

    def forward(self, images_in, masks_in, ws, noise_mode="random", return_stg1=False):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4].to(x.device)
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True).to(x.device)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode="bilinear", align_corners=False
        ).to(x.device)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        gs = self.to_style(fea_16).to(x.device)

        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode).to(x.device)

        img = img * (1 - masks_in) + images_in * masks_in

        if not return_stg1:
            return img
        else:
            return img, out_stg1

class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim, 
        w_dim,
        img_resolution,
        img_channels,
        synthesis_kwargs={},
        mapping_kwargs={},
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.mapping = MappingNet(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_layers,
            **mapping_kwargs,
        )

    def forward(
        self,
        images_in,
        masks_in,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
        noise_mode="none",
        return_stg1=False,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update,
        )
        img = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        return img

class MAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_arch = "MAT"

        self.model = Generator(
            z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3
        )
        self.z = torch.from_numpy(np.random.randn(1, self.model.z_dim))
        self.label = torch.zeros([1, self.model.c_dim])

    def forward(self, image, mask):
        image = image * 2 - 1
        mask = 1 - mask

        output = self.model(image, mask, self.z, self.label, truncation_psi=1, noise_mode="none")

        return output * 0.5 + 0.5

def mat_load(state_dict):
    state = {
        k.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping"): v
        for k, v in state_dict.items()
    }

    model = MAT()
    model.load_state_dict(state)
    return model

class LoadInpaintModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("inpaint"),),
            },
        }

    RETURN_TYPES = ("INPAINT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "AE.Tools/Inpaint"

    def load(self, model):
        from spandrel import ModelLoader

        model_file = folder_paths.get_full_path("inpaint", model)
        if model_file is None:
            raise RuntimeError(f"Model file not found: {model}")

        if model_file.endswith(".pt"):
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()
        else:
            sd = comfy.utils.load_torch_file(model_file, safe_load=True)

        if "synthesis.first_stage.conv_first.conv.resample_filter" in sd:
            model = mat_load(sd)
        else:
            model = ModelLoader().load_from_state_dict(sd)

        model = model.eval()
        return (model,)

def mask_unsqueeze(mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def to_torch(image, mask):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    if mask is not None:
        mask = mask_unsqueeze(mask)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask

def pad_reflect_once(x, original_padding):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = F.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(additional_padding), mode="constant")
    return x

def resize_square(image, mask, size):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)

def mask_floor(mask, threshold = 0.99):
    return (mask >= threshold).to(mask.dtype)

def undo_resize_square(image, original_size):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]

def to_comfy(image):
    return image.permute(0, 2, 3, 1)

class InpaintWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("INPAINT_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "value_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "AE.Tools/Inpaint"

    def inpaint(self, model, image, mask, value_seed):
        if isinstance(model, MAT):
            required_size = 512
        elif model.architecture.id == "LaMa":
            required_size = 256
        else:
            raise ValueError(f"Unknown model_arch {type(model)}")

        image, mask = to_torch(image, mask)
        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

        image_device = image.device
        device = get_torch_device()
        model.to(device)
        batch_image = []
        pbar = ProgressBar(batch_size)

        for i in trange(batch_size):
            work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
            work_image, work_mask, original_size = resize_square(
                work_image, work_mask, required_size
            )
            work_mask = mask_floor(work_mask)

            torch.manual_seed(value_seed)
            work_image = model(work_image.to(device), work_mask.to(device))

            work_image.to(image_device)
            work_image = undo_resize_square(work_image.to(image_device), original_size)
            work_image = image[i] + (work_image - image[i]) * mask_floor(mask[i])

            batch_image.append(work_image)
            pbar.update(1)

        model.cpu()
        result = torch.cat(batch_image, dim=0)
        return (to_comfy(result),)

class VAEEncodeInpaintConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "negative", "latent",)
    FUNCTION = "encode"
    CATEGORY = "AE.Tools/Inpaint"

    def encode(self, positive, negative, image, mask, vae):
        inpaint_model_conditioning = InpaintModelConditioning()

        try:
            positive, negative, latent = inpaint_model_conditioning.encode(positive, negative, image, vae, mask, noise_mask=True)
        except TypeError:
            positive, negative, latent = inpaint_model_conditioning.encode(positive, negative, image, vae, mask)

        return (positive, negative, latent,)