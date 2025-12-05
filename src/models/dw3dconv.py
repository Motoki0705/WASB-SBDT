from typing import Sequence, Tuple, Optional

import torch
from torch import nn, Tensor


# ------------------------------------------------------------
#  Basic building blocks
# ------------------------------------------------------------

class PixelShuffle3D(nn.Module):
    """3D version of pixel shuffle (a.k.a. voxel shuffle).

    Rearranges channels of shape [B, C * rD * rH * rW, D, H, W]
    into spatial dimensions [B, C, D * rD, H * rH, W * rW].

    Args:
        upscale: Scale factor for (D, H, W). Can be int or (rD, rH, rW).
    """

    def __init__(self, upscale: int | Tuple[int, int, int]) -> None:
        super().__init__()
        if isinstance(upscale, int):
            upscale = (upscale, upscale, upscale)
        self.rD, self.rH, self.rW = upscale

    def forward(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        rD, rH, rW = self.rD, self.rH, self.rW
        r_prod = rD * rH * rW

        if c % r_prod != 0:
            raise ValueError(
                f"Channel dimension {c} must be divisible by "
                f"rD*rH*rW={r_prod} for PixelShuffle3D."
            )

        out_c = c // r_prod
        x = x.view(b, out_c, rD, rH, rW, d, h, w)
        # (B, C, rD, rH, rW, D, H, W) -> (B, C, D, rD, H, rH, W, rW)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        # merge shuffled spatial dims
        x = x.reshape(b, out_c, d * rD, h * rH, w * rW)
        return x


class DWConv3dBlock(nn.Module):
    """Depthwise-separable 3D convolution block.

    depthwise Conv3d (groups=in_channels) -> pointwise Conv3d(1x1x1) -> BN -> ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Single int or (kD, kH, kW).
        padding: Optional padding; if None, uses "same" padding for the kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        padding: Optional[int | Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size

        if padding is None:
            padding = (kD // 2, kH // 2, kW // 2)

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(kD, kH, kW),
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DWConv2dBlock(nn.Module):
    """Depthwise-separable 2D convolution applied frame-wise.

    Expects input [B, C, T, H, W] and applies depthwise+pointwise Conv2d
    to each frame independently (no temporal mixing).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: 2D kernel size (int).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        # [B, C, T, H, W] -> [B*T, C, H, W]
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2d = self.depthwise(x2d)
        x2d = self.pointwise(x2d)
        x2d = self.bn(x2d)
        x2d = self.act(x2d)
        # back to [B, C_out, T, H, W]
        _, c_out, h_out, w_out = x2d.shape
        x3d = x2d.view(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4)
        return x3d


def center_crop_3d(x: Tensor, target_dhw: Tuple[int, int, int]) -> Tensor:
    """Center-crop [B, C, D, H, W] tensor to target (D, H, W) size."""
    _, _, d, h, w = x.shape
    td, th, tw = target_dhw

    sd = max((d - td) // 2, 0)
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)

    return x[:, :, sd:sd + td, sh:sh + th, sw:sw + tw]


# ------------------------------------------------------------
#  Encoder stages
# ------------------------------------------------------------

class EncoderStage2D(nn.Module):
    """Encoder stage that uses only 2D depthwise-separable convs (no temporal mixing).

    Structure:
        x -> (DWConv2dBlock × depth) -> skip
          -> MaxPool3d(kernel=(1,2,2)) -> x_down

    Input / output shapes:
        x, skip:    [B, C_out, T, H,   W  ]
        x_down:     [B, C_out, T, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(depth):
            blocks.append(DWConv2dBlock(ch_in, out_channels))
            ch_in = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.blocks(x)
        skip = x
        x_down = self.pool(x)
        return skip, x_down


class EncoderStage3D(nn.Module):
    """Encoder stage that uses 3D depthwise-separable convs (spatiotemporal).

    Structure:
        x -> (DWConv3dBlock × depth) -> skip
          -> MaxPool3d(kernel=2) -> x_down

    Input / output shapes:
        x, skip: [B, C_out, T,   H,   W  ]
        x_down:  [B, C_out, T/2, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(depth):
            blocks.append(DWConv3dBlock(ch_in, out_channels))
            ch_in = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.blocks(x)
        skip = x
        x_down = self.pool(x)
        return skip, x_down


# ------------------------------------------------------------
#  Decoder stages
# ------------------------------------------------------------

class DecoderStage(nn.Module):
    """Decoder stage with PixelShuffle3D upsampling and skip connection.

    Args:
        in_channels:  Channels of decoder input.
        skip_channels: Channels of encoder skip tensor.
        out_channels: Channels after this stage.
        depth:        Number of DWConv3dBlocks after concatenation.
        mode:
            "full"    -> upsample (T, H, W) by 2 (PixelShuffle3D(2,2,2)).
            "spatial" -> upsample (H, W) by 2 only (PixelShuffle3D(1,2,2));
                         conv kernels are (1,3,3) to avoid temporal mixing here.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in {"full", "spatial"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

        if mode == "full":
            upscale = (2, 2, 2)
            kernel = (3, 3, 3)
        else:
            upscale = (1, 2, 2)
            kernel = (1, 3, 3)

        rD, rH, rW = upscale
        self.pre_upsample = nn.Conv3d(
            in_channels,
            out_channels * rD * rH * rW,
            kernel_size=1,
            bias=False,
        )
        self.shuffle = PixelShuffle3D(upscale)

        # conv blocks after concatenation with skip
        blocks: list[nn.Module] = []
        ch_in = out_channels + skip_channels
        for _ in range(depth):
            blocks.append(DWConv3dBlock(ch_in, out_channels, kernel_size=kernel))
            ch_in = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # upsample
        x = self.pre_upsample(x)
        x = self.shuffle(x)

        # align shape with skip (center crop if necessary)
        if x.shape[2:] != skip.shape[2:]:
            x = center_crop_3d(x, skip.shape[2:])

        # concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        x = self.blocks(x)
        return x


# ------------------------------------------------------------
#  Top-level UNet
# ------------------------------------------------------------

class SpatioTemporalUNet(nn.Module):
    """UNet-style architecture with 2D shallow encoder and 3D deep encoder.

    - Input  : [B, T, C_in, H, W]
    - Output : [B, T, C_out, H, W]

    Design:
        * The first `num_spatial_stages` stages use 2D depthwise-separable convs
          (frame-wise, no temporal mixing) and spatial downsampling only.
        * The remaining stages use 3D depthwise-separable convs with 3D downsampling.
        * The decoder mirrors the encoder with PixelShuffle3D-based upsampling:
            - stages corresponding to 3D encoder blocks: upsample T/H/W (2x).
            - stages corresponding to 2D encoder blocks: upsample H/W only (2x).

    Note:
        For the default configuration (depths=[2,2,8,2]), the temporal dimension T
        should be divisible by 4 to cleanly down/up-sample twice.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        depths: Sequence[int] = (2, 2, 8, 2),
        channels: Sequence[int] = (32, 64, 128, 256),
        num_spatial_stages: Optional[int] = None,
        bottleneck_depth: int = 2,
    ) -> None:
        super().__init__()

        if len(depths) != len(channels):
            raise ValueError("depths and channels must have the same length.")
        self.num_stages = len(depths)
        if num_spatial_stages is None:
            # by default: last 2 stages are spatiotemporal
            num_spatial_stages = max(self.num_stages - 2, 0)
        if not (0 <= num_spatial_stages <= self.num_stages):
            raise ValueError("num_spatial_stages must be in [0, num_stages].")
        self.num_spatial_stages = num_spatial_stages

        # ---------------- Encoder ----------------
        encoders: list[nn.Module] = []
        prev_ch = in_channels
        for idx in range(self.num_stages):
            out_ch = channels[idx]
            depth = depths[idx]
            if idx < num_spatial_stages:
                stage = EncoderStage2D(prev_ch, out_ch, depth)
            else:
                stage = EncoderStage3D(prev_ch, out_ch, depth)
            encoders.append(stage)
            prev_ch = out_ch
        self.encoders = nn.ModuleList(encoders)

        # ---------------- Bottleneck ----------------
        bottleneck_ch = channels[-1] * 2
        blocks: list[nn.Module] = []
        ch_in = prev_ch
        for _ in range(bottleneck_depth):
            blocks.append(DWConv3dBlock(ch_in, bottleneck_ch))
            ch_in = bottleneck_ch
        self.bottleneck = nn.Sequential(*blocks)

        # ---------------- Decoder ----------------
        decoders: list[nn.Module] = []
        current_ch = bottleneck_ch
        for enc_idx in reversed(range(self.num_stages)):
            skip_ch = channels[enc_idx]
            out_ch = channels[enc_idx]
            depth = depths[enc_idx]
            if enc_idx < num_spatial_stages:
                mode = "spatial"
            else:
                mode = "full"
            decoders.append(
                DecoderStage(
                    in_channels=current_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    depth=depth,
                    mode=mode,
                )
            )
            current_ch = out_ch
        self.decoders = nn.ModuleList(decoders)

        # ---------------- Output head ----------------
        self.out_conv = nn.Conv3d(current_ch, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, C_in, H, W].

        Returns:
            Tensor of shape [B, T, C_out, H, W].
        """
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        skips: list[Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.out_conv(x)
        # back to [B, T, C_out, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return x


# ------------------------------------------------------------
#  WASB wrapper
# ------------------------------------------------------------

class DW3DConvForWASB(SpatioTemporalUNet):
    """Wrapper for WASB framework compatibility.

    Converts output format to dict of heatmaps: {0: heatmap}.
    Input: [B, T, C_in, H, W]
    Output: {0: [B, T, C_out, H, W]}
    """

    def forward(self, x: Tensor) -> dict[int, Tensor]:
        out = super().forward(x)
        return {0: out}


# ------------------------------------------------------------
#  Minimal usage example (for sanity check)
# ------------------------------------------------------------
if __name__ == "__main__":
    B, T, C_in, H, W = 8, 16, 3, 128, 128
    model = SpatioTemporalUNet(
        in_channels=C_in,
        out_channels=1,
        depths=(2, 2, 8, 2),
        channels=(32, 64, 128, 256),
        num_spatial_stages=2,
    ).to("cuda")
    for _ in range(10):
        x = torch.randn(B, T, C_in, H, W).to("cuda")
        y = model(x)
        print("output shape:", y.shape)  # expected: [2, 16, 1, 128, 128]
