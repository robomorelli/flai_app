"""
Standalone UCNN architecture for inference.

Pure PyTorch re-implementation of ``src.models.UCNN.UCNNSeq`` (flai training repo)
restricted to the configuration actually used by the exported checkpoints
(no cross-attention context). Parameter names match the Lightning state_dict,
so weights load with ``strict=True``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"Relu": nn.ReLU, "Elu": nn.ELU, "Tanh": nn.Tanh}


def conv_block(in_f, out_f, kernel_size, dilation, padding, padding_mode, activation, batch_norm):
    layers: List[nn.Module] = [
        nn.Conv1d(in_f, out_f, kernel_size, padding=padding, padding_mode=padding_mode, dilation=dilation)
    ]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))
    layers.append(activation())
    return nn.Sequential(*layers)


def deconv_block(in_f, out_f, kernel_size, padding, activation, batch_norm):
    # stride=1 transposed conv, as in the training code (deconv_block1D called with stride=1)
    layers: List[nn.Module] = [nn.ConvTranspose1d(in_f, out_f, kernel_size, 1, padding=padding)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_f))
    layers.append(activation())
    return nn.Sequential(*layers)


class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn, _ = self.self_attn(x, x, x)
        return self.norm(x + attn)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, padding_mode, activation, bn, pool_ks):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv1 = conv_block(in_ch, out_ch, kernel_size, dilation, padding, padding_mode, activation, bn)
        self.conv2 = conv_block(out_ch, out_ch, kernel_size, dilation, padding, padding_mode, activation, bn)
        self.pool = nn.MaxPool1d(kernel_size=pool_ks, stride=pool_ks)
        self.use_residual = in_ch == out_ch
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x if self.use_residual else self.residual_conv(x)
        x = self.conv2(self.conv1(x)) + identity
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation, bn, upsample_factor):
        super().__init__()
        self.up = upsample_factor
        padding = (kernel_size - 1) // 2
        self.conv1 = deconv_block(in_ch, out_ch, kernel_size, padding, activation, bn)
        self.conv2 = deconv_block(out_ch * 2, out_ch, kernel_size, padding, activation, bn)
        self.conv3 = deconv_block(out_ch, out_ch, kernel_size, padding, activation, bn)
        self.residual_conv = nn.Conv1d(out_ch * 2, out_ch, kernel_size=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.up, mode="nearest")
        if x.size(-1) != skip.size(-1):
            diff = abs(x.size(-1) - skip.size(-1))
            if x.size(-1) > skip.size(-1):
                skip = F.pad(skip, (diff // 2, diff // 2))
            else:
                x = F.pad(x, (diff // 2, diff // 2))
        if x.size(-1) != skip.size(-1):
            n = min(x.size(-1), skip.size(-1))
            x, skip = x[:, :, :n], skip[:, :, :n]
        x = self.conv1(x)
        x_cat = torch.cat((x, skip), dim=1)
        return self.conv3(self.conv2(x_cat)) + self.residual_conv(x_cat)


class BottleNeck(nn.Module):
    def __init__(self, in_f, out_f, kernel_size, padding, activation, bn, self_attention=False, num_heads=4):
        super().__init__()
        self.bottleneck = conv_block(in_f, out_f, kernel_size, 1, padding, "zeros", activation, bn)
        self.self_attention = SelfAttention1D(out_f, num_heads) if self_attention else None
        self.need_projection = self_attention and in_f != out_f
        if self.need_projection:
            self.residual_conv = nn.Conv1d(in_f, out_f, kernel_size=1)

    def forward(self, x):
        out = self.bottleneck(x)
        if self.self_attention is None:
            return out
        out = self.self_attention(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out + (self.residual_conv(x) if self.need_projection else x)


class SegmentClassifier(nn.Module):
    def __init__(self, in_channels, segment_length, num_classes):
        super().__init__()
        self.pool = nn.AvgPool1d(segment_length, stride=segment_length)
        self.conv = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(self.pool(x))


class UCNN(nn.Module):
    """Input [B, T, C] normalized signals -> logits [B, T // output_steps, n_classes]."""

    def __init__(self, n_features: int, n_classes: int, hps: Dict[str, Any]):
        super().__init__()
        if hps.get("CONTEXT_SIZE", 0):
            raise ValueError("Checkpoints with CONTEXT_SIZE > 0 (cross-attention) are not supported")
        if hps.get("BOTTLENECK_CROSSATTENTION", False):
            raise ValueError("Bottleneck cross-attention is not supported")

        act = ACTIVATIONS[hps["ACTIVATION"]]
        bn = hps["BN"]
        n_blocks = hps["NLAYERS"]
        base = hps["BASECHANNELS"]
        ks = hps["KERNEL_SIZE"]
        static = hps["STATIC"]
        dilation = hps["DILATION"]
        pool_ks = list(hps["POOL_KS"])
        if len(pool_ks) > n_blocks:
            pool_ks = pool_ks[:n_blocks]
        elif len(pool_ks) < n_blocks:
            pool_ks = pool_ks + pool_ks[-1:] * (n_blocks - len(pool_ks))
        self.output_steps = hps["OUTPUT_STEPS"]
        self.bottleneck_units = hps.get("BOTTLENECK_UNITS", 1)

        self.encoder = nn.ModuleList()
        for i in range(n_blocks):
            enc_in = n_features if i == 0 else (base if static else base * 2 ** (i - 1))
            enc_out = base if static else base * 2 ** i
            dil = dilation[i] if isinstance(dilation, list) else dilation
            self.encoder.append(EncoderBlock(enc_in, enc_out, ks, dil, hps["PADDING_MODE"], act, bn, pool_ks[i]))

        b_in, b_out = (base, base) if static else (base * 2 ** (n_blocks - 1), base * 2 ** n_blocks)
        self.bottleneck = BottleNeck(b_in, b_out, ks, (ks - 1) // 2, act, bn,
                                     self_attention=hps.get("BOTTLENECK_SELFATTENTION", False),
                                     num_heads=hps.get("NUM_HEADS", 4))

        self.decoder = nn.ModuleList()
        up = pool_ks[::-1]
        for i in range(n_blocks):
            dec_in = base if static else base * 2 ** (n_blocks - i)
            dec_out = base if static else base * 2 ** (n_blocks - i - 1)
            self.decoder.append(DecoderBlock(dec_in, dec_out, ks, act, bn, up[i]))

        self.segment_classifier = SegmentClassifier(base, self.output_steps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        skips = []
        for enc in self.encoder:
            x_full, x = enc(x)
            skips.append(x_full)
        for _ in range(self.bottleneck_units):
            x = self.bottleneck(x)
        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[-(i + 1)])
        return self.segment_classifier(x).permute(0, 2, 1)


def build_model(meta: Dict[str, Any]) -> UCNN:
    if meta["model"] != "UCNN":
        raise ValueError(f"Unsupported model type: {meta['model']}")
    return UCNN(n_features=len(meta["sensors"]), n_classes=len(meta["classes"]), hps=meta["hps"])
