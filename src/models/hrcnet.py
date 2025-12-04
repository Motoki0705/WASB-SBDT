from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthwiseBasicBlock(nn.Module):
    """Depthwise separable version of BasicBlock.

    Interface is compatible with BasicBlock:
        - inplanes: input channels
        - planes: output channels
        - stride: spatial stride for the first conv
        - downsample: optional residual projection

    Note:
        As with BasicBlock, if stride != 1 or inplanes != planes,
        the caller is responsible for providing `downsample`.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()

        # 1st depthwise + pointwise
        self.dw_conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=inplanes,
            bias=False,
        )
        self.pw_conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        # 2nd depthwise + pointwise
        self.dw_conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.pw_conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.dw_conv1(x)
        out = self.pw_conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dw_conv2(out)
        out = self.pw_conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


_BLOCKS = {
    "BASIC": BasicBlock,
    "DW_BASIC": DepthwiseBasicBlock,
    "BOTTLENECK": Bottleneck,
}


class TemporalModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityTemporal(TemporalModule):
    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TSMTemporal(TemporalModule):
    def __init__(self, channels: int, shift_div: int = 8, **kwargs: Any) -> None:
        super().__init__()
        self.channels = channels
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        fold = c // self.shift_div
        if fold == 0:
            return x

        out = torch.zeros_like(x)
        # shift part of channels backward (to t-1)
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # shift part of channels forward (to t+1)
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
        # remaining channels stay in place
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return out


class TemporalConv1D(TemporalModule):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        dilation: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = channels
        padding = (kernel_size - 1) // 2 * dilation
        for _ in range(max(num_layers, 1)):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=in_channels,
                    dilation=dilation,
                    bias=False,
                )
            )
            layers.append(nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x_tmp = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, c, t)
        x_tmp = self.conv(x_tmp)
        x_tmp = x_tmp.view(b, h, w, c, t).permute(0, 4, 3, 1, 2)
        return x_tmp


class ConvGRUTemporal(TemporalModule):
    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.channels = channels
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(
            channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_h = nn.Conv2d(
            channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        if hidden_channels != channels:
            self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        else:
            self.proj_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        h_state: Optional[torch.Tensor] = None
        outputs = []
        for i in range(t):
            x_t = x[:, i]
            if h_state is None:
                h_state = torch.zeros(
                    b,
                    self.hidden_channels,
                    h,
                    w,
                    device=x.device,
                    dtype=x.dtype,
                )
            combined = torch.cat([x_t, h_state], dim=1)
            zr = self.conv_zr(combined)
            z, r = torch.sigmoid(zr.chunk(2, dim=1))
            combined_r = torch.cat([x_t, r * h_state], dim=1)
            h_tilde = torch.tanh(self.conv_h(combined_r))
            h_state = (1.0 - z) * h_state + z * h_tilde
            outputs.append(h_state.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        if self.proj_out is not None:
            out = self.proj_out(out.view(b * t, self.hidden_channels, h, w)).view(
                b, t, self.channels, h, w
            )
        return out


class TransformerTemporal(TemporalModule):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dim_ff: Optional[int] = None,
        dropout: float = 0.1,
        depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        d_model = channels
        if dim_ff is None:
            dim_ff = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **kwargs,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x_tmp = x.permute(0, 3, 4, 1, 2)
        seq = x_tmp.reshape(b * h * w, t, c)
        seq = self.encoder(seq)
        x_tmp = seq.view(b, h, w, t, c)
        x = x_tmp.permute(0, 3, 4, 1, 2)
        return x


def build_temporal_module(
    temporal_type: str,
    channels: int,
    **kwargs: Any,
) -> TemporalModule:
    if temporal_type == "none":
        return IdentityTemporal(channels=channels, **kwargs)
    if temporal_type == "tsm":
        return TSMTemporal(channels=channels, **kwargs)
    if temporal_type == "conv1d":
        return TemporalConv1D(channels=channels, **kwargs)
    if temporal_type == "convgru":
        return ConvGRUTemporal(channels=channels, **kwargs)
    if temporal_type == "transformer":
        return TransformerTemporal(channels=channels, **kwargs)
    raise ValueError(f"Unknown temporal_type '{temporal_type}'.")

class HSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (F.relu6(x + 3.0, inplace=True) / 6.0)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu", "relu6"):
        return nn.ReLU(inplace=True)
    if name in ("hswish", "h-swish"):
        return HSwish()
    raise ValueError(f"Unknown activation '{name}'.")

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block used in MobileNetV3."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = _make_activation(activation)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale

class InvertedResidualV3(nn.Module):
    """MobileNetV3-style inverted residual block with optional SE."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: float = 4.0,
        use_se: bool = True,
        se_reduction: int = 4,
        activation: str = "hswish",
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}.")

        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1) and (in_ch == out_ch)

        layers: list[nn.Module] = []

        # 1x1 pointwise conv (expand)
        if hidden_dim != in_ch:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(hidden_dim, momentum=bn_momentum))
            layers.append(_make_activation(activation))

        # 3x3 depthwise conv
        layers.append(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(hidden_dim, momentum=bn_momentum))
        layers.append(_make_activation(activation))

        # SE
        if use_se:
            layers.append(
                SqueezeExcitation(
                    channels=hidden_dim,
                    reduction=se_reduction,
                    activation="relu",
                )
            )

        # 1x1 pointwise conv (project, linear)
        layers.append(
            nn.Conv2d(
                hidden_dim,
                out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_ch, momentum=bn_momentum))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out

class MobileNetV3Downsample(nn.Module):
    """Downsampling stack using MobileNetV3-style inverted residual blocks.

    Each block uses stride=2, so spatial size is reduced by 2**num_blocks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 4,
        expand_ratio: float = 4.0,
        use_se: bool = True,
        se_reduction: int = 4,
        activation: str = "hswish",
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1 for MobileNetV3Downsample.")

        layers: list[nn.Module] = []
        current_in = in_ch

        for i in range(num_blocks):
            # 最初のブロックで in_ch -> out_ch に合わせる
            current_out = out_ch
            layers.append(
                InvertedResidualV3(
                    in_ch=current_in,
                    out_ch=current_out,
                    stride=2,
                    expand_ratio=expand_ratio,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    activation=activation,
                    bn_momentum=bn_momentum,
                )
            )
            current_in = current_out

        self.down = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class ContextFusionBlock(nn.Module):
    """Two-resolution fusion block with repeated high/low interaction.

    This block updates a high-resolution spatiotemporal branch and a
    low-resolution spatiotemporal branch and lets them exchange information
    in both directions:

    1. High branch is updated by a stack of residual blocks.
    2. Low branch is updated by a stack of residual blocks and a transformer encoder.
    3. Low → High: low is upsampled and projected, then added to high.
    4. High → Low: high is pooled and projected, then added to low.
    The shapes must satisfy:

        high: (B, T, C_high, H, W)
        low : (B, T, C_low,  H_low, W_low)
    """

    def __init__(
        self,
        high_channels: int,
        low_channels: int,
        high_block: str = "DW_BASIC",
        low_block: str = "DW_BASIC",
        num_high_blocks: int = 2,
        num_low_blocks: int = 1,
        upsample_mode: str = "nearest",
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        temporal_type: str = "transformer",
        temporal_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if high_block not in _BLOCKS:
            raise ValueError(f"Unknown high_block '{high_block}'.")
        if low_block not in _BLOCKS:
            raise ValueError(f"Unknown low_block '{low_block}'.")

        high_block_cls = _BLOCKS[high_block]
        low_block_cls = _BLOCKS[low_block]

        self.high_path = nn.Sequential(
            *[
                high_block_cls(high_channels, high_channels)
                for _ in range(max(num_high_blocks, 1))
            ]
        )

        self.low_cnn = nn.Sequential(
            *[
                low_block_cls(low_channels, low_channels)
                for _ in range(max(num_low_blocks, 0))
            ]
        ) if num_low_blocks > 0 else nn.Identity()
        t_kwargs = dict(transformer_kwargs or {})
        d_model = t_kwargs.pop("d_model", low_channels)
        num_heads = t_kwargs.pop("num_heads", 8)
        dim_ff = t_kwargs.pop("dim_ff", d_model * 4)
        dropout = t_kwargs.pop("dropout", 0.1)
        depth = t_kwargs.pop("depth", 2)

        if d_model != low_channels:
            raise ValueError(
                f"d_model ({d_model}) must match low_channels ({low_channels})."
            )

        encoder_layer_spatial = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **t_kwargs,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer_spatial,
            num_layers=depth,
        )

        temporal_kwargs_all: Dict[str, Any] = {
            "num_heads": num_heads,
            "dim_ff": dim_ff,
            "dropout": dropout,
            "depth": depth,
            **t_kwargs,
        }
        if temporal_kwargs is not None:
            temporal_kwargs_all.update(temporal_kwargs)

        self.temporal = build_temporal_module(
            temporal_type=temporal_type,
            channels=low_channels,
            **temporal_kwargs_all,
        )

        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels, momentum=BN_MOMENTUM),
        )

        self.upsample_mode = upsample_mode
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fusion step.

        Args:
            high: High-resolution feature sequence,
                shape (B, T, C_high, H, W).
            low: Low-resolution feature sequence,
                shape (B, T, C_low, H_low, W_low).

        Returns:
            Tuple of updated (high, low) feature maps with the same shapes as inputs.
        """
        # Expect high: (B, T, C_high, H, W), low: (B, T, C_low, H_low, W_low)
        b, t, c_high, h, w = high.shape
        _, _, c_low, h_low, w_low = low.shape
        high_flat = high.view(b * t, c_high, h, w)
        low_flat = low.view(b * t, c_low, h_low, w_low)

        high_flat = self.high_path(high_flat)
        low_flat = self.low_cnn(low_flat)

        high = high_flat.view(b, t, c_high, h, w)
        low = low_flat.view(b, t, c_low, h_low, w_low)

        low_spatial = low.view(b * t, c_low, h_low * w_low).permute(0, 2, 1)
        low_spatial = self.spatial_transformer(low_spatial)
        low = low_spatial.permute(0, 2, 1).view(b, t, c_low, h_low, w_low)

        low = self.temporal(low)

        high_flat = high.view(b * t, c_high, h, w)
        low_flat = low.reshape(b * t, c_low, h_low, w_low)

        low_up = F.interpolate(low_flat, size=(h, w), mode=self.upsample_mode)
        low_up = self.low_to_high(low_up)
        high_flat = self.activation(high_flat + low_up)
        high = high_flat.view(b, t, c_high, h, w)

        pooled = F.adaptive_avg_pool2d(high_flat, output_size=(h_low, w_low))
        pooled = self.high_to_low(pooled)
        low_flat = self.activation(low_flat + pooled)
        low = low_flat.view(b, t, c_low, h_low, w_low)

        return high, low


class HRCNet(nn.Module):
    """High-Resolution Context Net (HRCNet).

    This architecture maintains a high-resolution convolutional branch and a
    low-resolution transformer branch. They exchange information multiple times
    in a stack of ContextFusionBlocks, in a spirit similar to HRNet but with
    only two resolutions and a transformer in the low-resolution branch.

    The overall flow is:

        input -> stem -> branch0 (high)
            branch0 -> MobileNetDownsample -> branch4 (low, 1/16 resolution)
            for each fusion stage:
                (branch0, branch4) = ContextFusionBlock(branch0, branch4)
        output head uses branch0 (high resolution).

    Only the high-resolution output is intended for downstream tasks, but the
    low-resolution features are kept for diagnostics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        high_channels: int,
        low_channels: int,
        num_stages: int = 3,
        high_block: str = "DW_BASIC",
        low_block: str = "DW_BASIC",
        num_high_blocks: int = 2,
        num_low_blocks: int = 1,
        upsample_mode: str = "nearest",
        downsample_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        temporal_type: str = "transformer",
        temporal_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if num_stages < 1:
            raise ValueError("num_stages must be >= 1.")

        self.in_channels = 3
        self.out_channels = 1
        self.high_channels = high_channels
        self.low_channels = low_channels
        self.num_stages = num_stages

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                high_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.initial_down = MobileNetV3Downsample(
            in_ch=high_channels,
            out_ch=low_channels,
            **(downsample_kwargs or {}),
        )

        stages: list[ContextFusionBlock] = []
        for _ in range(num_stages):
            stages.append(
                ContextFusionBlock(
                    high_channels=high_channels,
                    low_channels=low_channels,
                    high_block=high_block,
                    low_block=low_block,
                    num_high_blocks=num_high_blocks,
                    num_low_blocks=num_low_blocks,
                    upsample_mode=upsample_mode,
                    transformer_kwargs=transformer_kwargs,
                    temporal_type=temporal_type,
                    temporal_kwargs=temporal_kwargs,
                )
            )
        self.stages = nn.ModuleList(stages)

        self.head = nn.Conv2d(
            high_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution and normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the HRCNet forward pass.

        Args:
            x: Input tensor of shape (B, T, in_channels, H, W).

        Returns:
            Dict with:
                - "out": high-resolution output sequence,
                    shape (B, T, out_channels, H, W)
                - "high": final high-resolution features,
                    shape (B, T, C_high, H, W)
                - "low": final low-resolution features,
                    shape (B, T, C_low, H_low, W_low)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got shape {x.shape}.")

        b, t, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, but got C={c}.")

        x_flat = x.view(b * t, c, h, w)

        high_flat = self.stem(x_flat)
        low_flat = self.initial_down(high_flat)

        _, _, h_low, w_low = low_flat.shape

        high = high_flat.view(b, t, self.high_channels, high_flat.shape[-2], high_flat.shape[-1])
        low = low_flat.view(b, t, self.low_channels, h_low, w_low)

        for stage in self.stages:
            high, low = stage(high, low)

        # High branch back to (B*T, C_high, H, W) for the head
        _, _, c_high, h_out, w_out = high.shape
        high_flat = high.view(b * t, c_high, h_out, w_out)
        out_flat = self.head(high_flat)
        out = out_flat.view(b, t, self.out_channels, h_out, w_out)

        return {
            "out": out,
            "high": high,
            "low": low,
        }


class HRCNetForWASB(HRCNet):
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        outputs = super().forward(x)
        return {0: outputs["out"]}