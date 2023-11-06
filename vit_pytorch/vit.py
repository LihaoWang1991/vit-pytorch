import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):   # dim, heads, dim_head, dropout: 1024, 16, 64, 0.1
        super().__init__()
        inner_dim = dim_head *  heads   # 64 * 16 = 1024
        project_out = not (heads == 1 and dim_head == dim)    # True

        self.heads = heads
        self.scale = dim_head ** -0.5   # 0.125 (1 / 8)

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)   # 2, 65, 1024 (B, 64 + 1, dim)

        qkv = self.to_qkv(x).chunk(3, dim = -1)   # tuple of length 3, each of shape (2, 65, 1024)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)   # 2, 16, 65, 64

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # 2, 16, 65, 65

        attn = self.attend(dots)   # 2, 16, 65, 65
        attn = self.dropout(attn)  # 2, 16, 65, 65

        out = torch.matmul(attn, v)   # 2, 16, 65, 64
        out = rearrange(out, 'b h n d -> b n (h d)')   # 2, 65, 1024
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):   # dim, depth, heads, dim_head, mlp_dim, dropout: 1024, 6, 16, 64, 2048, 0.1
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # 2, 65, 1024
            x = ff(x) + x     # 2, 65, 1024

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)   # 8 * 8 = 64
        patch_dim = channels * patch_height * patch_width    # 3 * 32 * 32 = 3072
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))    # 1, 65, 1024 (B, 64 + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   # 1, 1, 1024
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)    # 2, 64, 1024 (B, num_patches, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)    # 2, 1, 1024 (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)    # 2, 65, 1024 (B, 64 + 1, dim)
        x += self.pos_embedding[:, :(n + 1)]    # 2, 65, 1024
        x = self.dropout(x)   # 2, 65, 1024

        x = self.transformer(x)    # 2, 65, 1024

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]    # 2, 1024

        x = self.to_latent(x)    # 2, 1024
        return self.mlp_head(x)   # 2, 1000
