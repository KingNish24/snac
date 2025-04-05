import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F # Added F

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantize


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
        # --- NEW Parameters for High Retention ---
        use_global_skip=True,
        use_latent_bypass=True,
        initial_latent_bypass_scale=0.05, # Start small
        # -----------------------------------------
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.total_stride = self.hop_length # Same as hop_length

        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )

        # --- MODIFIED Decoder Instantiation ---
        self.decoder = Decoder(
            latent_dim, # Input is still based on latent dim for main path
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
            # Pass skip connection info
            use_global_skip=use_global_skip,
            global_skip_channels=decoder_dim // 4 # Example: project skip to fewer channels initially
        )
        # -----------------------------------

        # --- NEW Parameters/Layers for High Retention ---
        self.use_global_skip = use_global_skip
        self.use_latent_bypass = use_latent_bypass

        if self.use_latent_bypass:
            # Learnable scale for how much unquantized latent bypasses VQ
            self.latent_bypass_scale = nn.Parameter(torch.tensor(float(initial_latent_bypass_scale)))

        if self.use_global_skip:
             # Layer to project downsampled input to match decoder's skip input channels
            self.global_skip_projection = nn.Conv1d(
                1, self.decoder.global_skip_channels, kernel_size=7, padding=3, bias=False
            )
            # Optional: Add learnable scaling for the global skip as well
            # self.global_skip_scale = nn.Parameter(torch.tensor(1.0))
        # --------------------------------------------

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        # Ensure length is divisible by total stride for skip connection matching
        lcm_stride = math.lcm(self.total_stride, self.hop_length * (self.vq_strides[0] if self.vq_strides else 1))
        lcm = math.lcm(lcm_stride, self.attn_window_size or 1)
        pad_to = math.ceil(length / lcm) * lcm
        right_pad = pad_to - length
        # Pad input symmetrically if possible, or use functional.pad
        # audio_data = nn.functional.pad(audio_data, (0, right_pad)) # Original
        # Symmetric padding might be slightly better for convs, but functional is fine
        audio_data = nn.functional.pad(audio_data, (right_pad // 2, right_pad - right_pad // 2))

        return audio_data, length # Return original length too

    def forward(self, audio_data: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        original_length = audio_data.shape[-1]
        # Preprocess returns padded audio and original length
        audio_data_padded, _ = self.preprocess(audio_data)

        # --- Global Skip Path ---
        global_skip_info = None
        if self.use_global_skip:
            # Downsample input audio to match the temporal resolution *before* decoder blocks
            # The target length should match z's length.
            target_length = audio_data_padded.shape[-1] // self.total_stride
            # Use adaptive avg pooling to handle potential rounding issues
            downsampled_input = F.adaptive_avg_pool1d(audio_data_padded, target_length)
            # Project channels
            global_skip_info = self.global_skip_projection(downsampled_input)
            # Optional scaling: global_skip_info = self.global_skip_scale * global_skip_info
        # ------------------------

        # --- Main Autoencoder Path ---
        # 1. Encode
        z = self.encoder(audio_data_padded) # z shape: (B, latent_dim, T_latent)

        # 2. Quantize
        z_q, codes = self.quantizer(z) # z_q shape: (B, latent_dim, T_latent)

        # 3. Prepare Decoder Input (Potentially bypassing VQ)
        if self.use_latent_bypass:
            # Combine quantized latent with original latent
            # Ensure z and z_q have compatible shapes (should be the case here)
            z_decoder_input = z_q + self.latent_bypass_scale * z
        else:
            # Use only the quantized latent
            z_decoder_input = z_q
        # ---------------------------

        # 4. Decode (Pass global skip info if used)
        audio_hat_padded = self.decoder(z_decoder_input, global_skip=global_skip_info)

        # Trim padding to match original input length
        pad_amount_start = (audio_hat_padded.shape[-1] - original_length) // 2
        pad_amount_end = audio_hat_padded.shape[-1] - original_length - pad_amount_start
        audio_hat = audio_hat_padded[..., pad_amount_start:audio_hat_padded.shape[-1]-pad_amount_end]

        return audio_hat, codes

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        # Encoding should remain the same - just produce codes
        audio_data_padded, _ = self.preprocess(audio_data)
        z = self.encoder(audio_data_padded)
        _, codes = self.quantizer(z) # Use the standard quantizer forward pass
        return codes

    def decode(self, codes: List[torch.Tensor], reference_length: int = None) -> torch.Tensor:
        # Decoding now needs potential mechanisms if bypass was used,
        # but for standard decoding from codes, bypass isn't available.
        # We *only* use the quantized path here.
        z_q = self.quantizer.from_codes(codes)

        # --- Need a way to handle global skip during pure decode ---
        # Option 1: Don't use global skip during pure decode (pass None)
        # Option 2: Generate a dummy skip (e.g., zeros) - less ideal
        # Let's go with Option 1 for simplicity. The high-retention is
        # primarily for the end-to-end pass or analysis of the untrained state.
        global_skip_info = None
        # If we *needed* to approximate skip during decode, we might need
        # to generate noise or a base signal of the right shape, but let's skip that complexity.

        audio_hat_padded = self.decoder(z_q, global_skip=global_skip_info)

        # If reference length is provided, trim. Otherwise, return padded.
        if reference_length is not None:
             # Estimate padding based on hop_length if needed, but using reference_length is better
            pad_amount_start = (audio_hat_padded.shape[-1] - reference_length) // 2
            pad_amount_end = audio_hat_padded.shape[-1] - reference_length - pad_amount_start
            # Handle potential off-by-one from padding calculation if length isn't perfectly divisible
            if pad_amount_start < 0 or pad_amount_end < 0:
                 # Fallback or error, maybe just return center crop based on length
                 current_len = audio_hat_padded.shape[-1]
                 start = max(0, (current_len - reference_length) // 2)
                 audio_hat = audio_hat_padded[..., start:start + reference_length]
            else:
                audio_hat = audio_hat_padded[..., pad_amount_start:audio_hat_padded.shape[-1]-pad_amount_end]

            # Final check for exact length if reference_length was provided
            if audio_hat.shape[-1] != reference_length:
                 # Simple truncation / padding if needed due to rounding
                 if audio_hat.shape[-1] > reference_length:
                     audio_hat = audio_hat[..., :reference_length]
                 else:
                     diff = reference_length - audio_hat.shape[-1]
                     audio_hat = F.pad(audio_hat, (0, diff)) # Pad end
        else:
            audio_hat = audio_hat_padded # Return potentially padded output

        return audio_hat

    # Keep from_config and from_pretrained, but note that saved models
    # might not have the bypass parameters if trained with older code.
    # Need to handle loading state_dict carefully (strict=False potentially).
    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Ensure new parameters have defaults if loading old config
        config.setdefault('use_global_skip', False) # Default to False for old configs
        config.setdefault('use_latent_bypass', False)
        config.setdefault('initial_latent_bypass_scale', 0.0)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        from huggingface_hub import hf_hub_download
        strict_loading = kwargs.pop('strict_loading', False) # Allow non-strict loading

        if not os.path.isdir(repo_id):
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
            model = cls.from_config(config_path)
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            model = cls.from_config(os.path.join(repo_id, "config.json"))
            state_dict = torch.load(os.path.join(repo_id, "pytorch_model.bin"), map_location="cpu")

        # Load state dict potentially non-strictly
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_loading)
        if not strict_loading:
             if missing_keys: print(f"Warning: Missing keys during loading: {missing_keys}")
             if unexpected_keys: print(f"Warning: Unexpected keys during loading: {unexpected_keys}")

        model.eval()
        return model
