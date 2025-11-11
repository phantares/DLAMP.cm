import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, q_channel, kv_channel, token_channel=128, heads=4):
        super().__init__()

        self.heads = heads
        self.d_heads = token_channel // heads
        self.scale = self.d_heads**-0.5
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv3d(q_channel, token_channel, 1)
        self.k = nn.Linear(kv_channel, token_channel, bias=False)
        self.v = nn.Linear(kv_channel, token_channel, bias=False)
        self.proj = nn.Conv3d(token_channel, q_channel, 1)

        self.ff = nn.Sequential(
            nn.Conv3d(q_channel, q_channel * 2, 1),
            nn.SiLU(),
            nn.Conv3d(q_channel * 2, q_channel, 1),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat, tokens):  # feat: (B,C,D,H,W), tokens: (B,T,c_tok)
        _, _, Z, H, W = feat.shape

        q = rearrange(
            self.q(feat),
            "b (nh dh) z h w -> b nh (z h w) dh",
            nh=self.heads,
            dh=self.d_heads,
        )  # (B,heads,Z*H*W,d_head)
        k = rearrange(
            self.k(tokens), "b n (nh dh) -> b nh n dh", nh=self.heads, dh=self.d_heads
        )  # (B,heads,n,d_head)
        v = rearrange(
            self.v(tokens), "b n (nh dh) -> b nh n dh", nh=self.heads, dh=self.d_heads
        )

        attn = self.softmax((q @ k.transpose(-1, -2)) * self.scale)  # (B,heads,Z*H*W,n)
        output = attn @ v  # (B,heads,Z*H*W,d_head)

        output = rearrange(output, "b nh (z h w) dh -> b (nh dh) z h w", z=Z, h=H, w=W)
        output = self.proj(output)

        output = feat + self.gate.tanh() * output
        output = output + self.ff(output)

        return output
