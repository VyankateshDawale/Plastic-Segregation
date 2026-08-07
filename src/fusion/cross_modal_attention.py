"""
cross_modal_attention.py — Deep cross-modal fusion prototype
============================================================
A self-contained, runnable prototype of *learnable* deep fusion of the three
modalities used by the sorting system:

    - RGB    : spatial feature map from the YOLO detection backbone
    - NIR    : 1-D spectral reflectance vector (e.g. 256 bands)
    - MIR    : 1-D spectral vector (e.g. 128 bands / FTIR)

The deployed system today fuses these with a *hard-coded sequential cascade*
(IF vision_conf < t THEN escalate to NIR; IF NIR_conf < 0.85 THEN escalate to
MIR). That logic is fixed and hand-tuned. This module replaces it with a
small bidirectional cross-attention block that lets the model *learn* how much
to weight each modality per-object — the second patentable contribution
beyond the confidence-gated escalation.

Design
------
    RGB features  : (B, C, H, W)  -> flatten/pool -> (B, N, D)
    Spectral      : (B, S)        -> MLP embed     -> (B, M, D)
    CrossAttention(RGB_queries, Spectral_keys/values) -> RGB attends to spectral
    CrossAttention(Spectral_queries, RGB_keys/values) -> Spectral attends to RGB
    Concatenate & pool -> fused vector -> MLP head -> class logits

The script runs a forward pass on random tensors so it can be verified
standalone (no GPU / no real sensors required). Replace the random tensors
with real backbone outputs + sensor readings to deploy.

Usage:
    python src/fusion/cross_modal_attention.py            # self-test forward pass
    python src/fusion/cross_modal_attention.py --batch 4  # different batch size
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """Bidirectional cross-attention fusion of RGB + spectral features."""

    def __init__(
        self,
        rgb_channels: int = 256,    # backbone feature channels (e.g. YOLO C3 output)
        spectral_dim: int = 384,    # NIR bands + MIR bands concatenated
        embed_dim: int = 128,       # shared attention dimension D
        n_heads: int = 4,
        num_classes: int = 6,      # e.g. BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC
        spatial_tokens: int = 64,   # number of RGB patch tokens kept after pooling
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_tokens = spatial_tokens

        # --- RGB projection: (B,C,H,W) -> (B, N, D) ---
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(rgb_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Adaptive pooling to a fixed number of spatial tokens
        self.rgb_pool = nn.AdaptiveAvgPool2d((8, 8))  # -> 64 tokens

        # --- Spectral projection: (B, S) -> (B, M, D) ---
        # We split the spectral vector into M pseudo-tokens so attention has
        # something to attend over; here M = 8 spectral "bands-of-interest".
        self.spectral_tokens = 8
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_dim, embed_dim * self.spectral_tokens),
            nn.ReLU(inplace=True),
        )
        self.spectral_reshape = lambda x: x.view(
            x.size(0), self.spectral_tokens, embed_dim
        )

        # --- Bidirectional cross-attention ---
        self.rgb_to_spectral = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.spectral_to_rgb = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # --- Fusion head ---
        # Total tokens after concat = spatial_tokens(64) + spectral_tokens(8)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * (spatial_tokens + self.spectral_tokens), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _rgb_to_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.rgb_pool(self.rgb_proj(rgb))          # (B, D, 8, 8)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)               # (B, h*w, D)
        return x[:, : self.spatial_tokens, :]          # (B, N, D)

    def _spectral_to_tokens(self, spectral: torch.Tensor) -> torch.Tensor:
        x = self.spectral_proj(spectral)               # (B, D*M)
        return self.spectral_reshape(x)                # (B, M, D)

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        # rgb: (B, C, H, W), spectral: (B, S)
        rgb_tok = self._rgb_to_tokens(rgb)             # (B, N, D)
        spec_tok = self._spectral_to_tokens(spectral)  # (B, M, D)

        # RGB queries attend to spectral keys/values  -> each patch sees spectral
        attn_out1, _ = self.rgb_to_spectral(rgb_tok, spec_tok, spec_tok)
        rgb_tok = self.norm1(rgb_tok + attn_out1)

        # Spectral queries attend to RGB keys/values -> spectral informed by scene
        attn_out2, _ = self.spectral_to_rgb(spec_tok, rgb_tok, rgb_tok)
        spec_tok = self.norm2(spec_tok + attn_out2)

        fused = torch.cat([rgb_tok, spec_tok], dim=1)  # (B, N+M, D)
        fused = fused.flatten(1)                       # (B, (N+M)*D)
        return self.head(fused)                        # (B, num_classes)


def self_test(batch_size: int = 2):
    """Run a forward pass on random tensors to verify the module is sound."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CrossModalFusion(
        rgb_channels=256, spectral_dim=384, num_classes=6
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fusion] CrossModalFusion params: {n_params:,}  device={device}")

    # Realistic-ish inputs
    rgb = torch.randn(batch_size, 256, 16, 16, device=device)   # backbone feature map
    nir = torch.randn(batch_size, 256, device=device)           # NIR reflectance
    mir = torch.randn(batch_size, 128, device=device)           # MIR spectrum
    spectral = torch.cat([nir, mir], dim=1)                    # (B, 384)

    out = model(rgb, spectral)
    probs = F.softmax(out, dim=1)

    print(f"[fusion] rgb {tuple(rgb.shape)}  spectral {tuple(spectral.shape)}")
    print(f"[fusion] logits {tuple(out.shape)}  probs {tuple(probs.shape)}")
    assert out.shape == (batch_size, 6), "output shape mismatch"
    assert torch.allclose(probs.sum(1), torch.ones(batch_size, device=device), atol=1e-5)
    print("[fusion] Forward pass OK - cross-attention fusion produces 6-class logits.")
    print("   Note: this replaces the fixed IF/ELSE NIR->MIR cascade with a")
    print("   learned, per-object weighting of RGB + NIR + MIR modalities.")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    self_test(args.batch)
