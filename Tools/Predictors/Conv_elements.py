import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange



#################### Residual layer ####################
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

#################### Activation layer ####################
class Activation(nn.Module):
    def __init__(self, name="gelu"):
        super().__init__()

        name = name.lower()

        if name == "relu":
            self.act = nn.ReLU()

        elif name == "gelu":
            self.act = nn.GELU()

        elif name == "gelu_tanh":
            self.act = nn.GELU(approximate="tanh")

        elif name == "silu":
            self.act = nn.SiLU()

        elif name == "elu":
            self.act = nn.ELU()

        elif name == "leakyrelu":
            self.act = nn.LeakyReLU(0.1)

        elif name == "tanh":
            self.act = nn.Tanh()

        elif name == "sigmoid":
            self.act = nn.Sigmoid()

        else:
            raise ValueError(f"Unknown activation {name}")

    def forward(self, x):
        return self.act(x)




#################### pooling layer ####################
# cite from lucidrains/enformer-pytorch
class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)


class MaxPool1D(nn.Module):
    def __init__(
        self, 
        kernel_size=2, 
        stride=2, 
        ceil_mode=False,

        padding=0, 
        dilation=1, 
        return_indices=False
        ):

        super().__init__()

        self.pool = nn.MaxPool1d(
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            return_indices=return_indices, 
            ceil_mode=ceil_mode
        )

    def forward(self, x):
        return self.pool(x)



class MeanPool1D(nn.Module):
    def __init__(
        self, 
        kernel_size=2, 
        stride=2, 
        ceil_mode=False, 

        padding=0, 
        count_include_pad=True
        ):

        super().__init__()

        self.pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

    def forward(self, x):
        return self.pool(x)



#################### various Blocks ####################
class ConvBlock1(nn.Module):
    # batchnorm + activation + conv + optional residual
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding="same",
        dilation=1,
        norm=True,
        activation="gelu",
        residual=False,
        bias=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        layers = []

        if norm:
            layers.append(nn.BatchNorm1d(in_channels))

        if activation is not None:
            layers.append(Activation(activation))

        self.pre = nn.Sequential(*layers)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # padding is controled by forward
            dilation=dilation,
            bias=bias,
        )

        # residual
        if residual:
            if in_channels != out_channels:
                self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
            else:
                self.residual_proj = nn.Identity()
        else:
            self.residual_proj = None


    def _same_padding(self, x):
        """
        the last legal kernel start is (L_in - kernel_size)
        L_out is ((L_in - kernel_size)/stride + 1)
        when add padding is L_out = ((L_in + P - kernel_size)/stride + 1)
        So P = (L_out - 1) * stride + kernel_size - L_in
        """
        L_in = x.shape[-1]
        L_out = (L_in + self.stride - 1) // self.stride  # ceil
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        pad_needed = max(
            0,
            (L_out - 1) * self.stride + effective_kernel - L_in
        )
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return pad_left, pad_right


    def forward(self, x):
        residual_input = x
        x = self.pre(x)

        if self.padding == "same":
            pad_left, pad_right = self._same_padding(x)
            if pad_left or pad_right:
                x = F.pad(x, (pad_left, pad_right))

        elif isinstance(self.padding, int):
            if self.padding > 0:
                x = F.pad(x, (self.padding, self.padding))

        
        out = self.conv(x)

        if self.residual_proj is not None:
            out = out + self.residual_proj(residual_input)

        return out