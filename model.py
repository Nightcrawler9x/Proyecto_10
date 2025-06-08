import torch
import torch.nn as nn
import math

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return x

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        return x

# Bloque Transformer simple (solo 1 capa, 1 cabeza, sin modulator, sin shift)
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=1, win_size=8, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.input_resolution = input_resolution

    def forward(self, x):
        # x: (B, N, C)
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class Uformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3, embed_dim=16, win_size=8):
        super().__init__()
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.encoder = SimpleTransformerBlock(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            num_heads=1,
            win_size=win_size,
            mlp_ratio=2.
        )

    def forward(self, x):
        y = self.input_proj(x)
        y = self.encoder(y)
        y = self.output_proj(y)
        return x + y  # Residual

# Ejemplo de uso:
# model = Uformer(img_size=128, in_chans=3, embed_dim=16, win_size=8)
