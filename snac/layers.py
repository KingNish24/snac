import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .attention import LocalMHA


class Encoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel, # This is the dimension of z or z_q
        channels,      # Target dimension after initial projection
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=32,
        d_out=1,
        # --- NEW Parameters for Global Skip ---
        use_global_skip=False,
        global_skip_channels=None # Channels expected from the projected global skip
        # -------------------------------------
    ):
        super().__init__()
        self.use_global_skip = use_global_skip
        self.global_skip_channels = global_skip_channels

        # --- Initial Projection Layers ---
        # Project the main latent input (z_q + bypass) to the main channel dimension
        if depthwise:
            # Keep original depthwise logic if desired, applied to the main latent path
            self.initial_proj = nn.Sequential(
                 WNConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
                 WNConv1d(input_channel, channels, kernel_size=1),
             )
        else:
             self.initial_proj = WNConv1d(input_channel, channels, kernel_size=7, padding=3)

        # Optional: Layer to combine projected latent + projected skip before attention/blocks
        if self.use_global_skip:
             # Ensure global_skip_channels was provided
            if global_skip_channels is None:
                raise ValueError("global_skip_channels must be provided if use_global_skip is True")
            # 1x1 Conv to merge the projected latent and the projected skip
            # The input channels here is the main 'channels' + 'global_skip_channels'
            self.merge_conv = WNConv1d(channels + global_skip_channels, channels, kernel_size=1)
        # ---------------------------------

        # --- Optional Attention (applied after potential merge) ---
        layers = []
        current_channels = channels # Channels after initial projection and merge
        if attn_window_size is not None:
            layers += [LocalMHA(dim=current_channels, window_size=attn_window_size)]
        # -----------------------------------------------------

        # --- Decoder Blocks (Upsampling) ---
        for i, stride in enumerate(rates):
            input_dim = current_channels // 2**i
            output_dim = current_channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            # Pass noise flag to DecoderBlock
            layers.append(DecoderBlock(input_dim, output_dim, stride, noise, groups=groups))
        # -----------------------------------

        # --- Final Output Layers ---
        final_input_dim = current_channels // 2 ** len(rates)
        layers += [
            Snake1d(final_input_dim),
            WNConv1d(final_input_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        # ---------------------------
        self.model_blocks = nn.Sequential(*layers) # Changed name to avoid clash

    def forward(self, x, global_skip=None): # x is z_decoder_input
        # 1. Initial projection of the main latent path
        x_proj = self.initial_proj(x)

        # 2. Merge with Global Skip (if enabled and provided)
        if self.use_global_skip and global_skip is not None:
            # Ensure temporal dimensions match (should be handled by SNAC's pooling)
            if x_proj.shape[-1] != global_skip.shape[-1]:
                 # This indicates an issue in preprocessing or pooling calculation
                 # Fallback: Adaptive pooling here, but it's better to fix upstream
                 print(f"Warning: Mismatch in decoder skip connection lengths: {x_proj.shape[-1]} vs {global_skip.shape[-1]}. Resizing.")
                 target_len = x_proj.shape[-1]
                 global_skip = F.adaptive_avg_pool1d(global_skip, target_len)
                 # Alternative: Pad the shorter one

            # Concatenate along the channel dimension
            merged = torch.cat([x_proj, global_skip], dim=1)
            # Merge channels back to the main dimension
            h = self.merge_conv(merged)
        elif self.use_global_skip and global_skip is None:
            # If skip is configured but not provided (e.g., during decode from codes),
            # create zero padding for the skip channels to maintain architecture.
            # This is arguably better than branching logic inside the main path.
            print("Warning: Decoder configured for global skip, but none provided. Using zeros.")
            zeros_skip = torch.zeros(
                x_proj.shape[0], self.global_skip_channels, x_proj.shape[2],
                device=x_proj.device, dtype=x_proj.dtype
            )
            merged = torch.cat([x_proj, zeros_skip], dim=1)
            h = self.merge_conv(merged)
        else:
            # No global skip used
            h = x_proj

        # 3. Pass through main decoder blocks (Attention -> Upsampling -> Final Layers)
        out = self.model_blocks(h)
        return out

class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
