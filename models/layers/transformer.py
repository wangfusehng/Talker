import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from einops import rearrange
from .config import use_fused_attn
from .helpers import to_2tuple
__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):  # Fixed method name with double underscores
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # Instead of a combined QKV projection, we have separate Q and KV projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x: Query input of shape (B, N, C)
            context: Key/Value input of shape (B, M, C)
        """
        B, N, C = x.shape
        M = context.shape[1]

        # Project queries from x
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Project keys and values from context
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Apply normalization if specified
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


# class CrossAttnGatedFusion(nn.Module):
#     def __init__(self, dim,num_heads):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.linear_b = nn.Linear(dim, dim)
#         self.linear_attn = nn.Linear(dim, dim)
#         self.gate = nn.Linear(dim * 2, dim)

#     def forward(self, audio_feat, motion_feat):
#         # audio_feat: (B, T1, D), motion_feat: (B, T2, D)
        
#         # Step 1: Cross-Attention (A → B)
#         # 将 audio 作为 key, value；motion 作为 query
#         attn_output, _ = self.cross_attn(query=motion_feat, key=audio_feat, value=audio_feat)  # (B, T2, D)
        
#         # Step 2: Gated Fusion between attn_output and original motion_feat
#         z1 = torch.tanh(self.linear_b(motion_feat))       # 原始 motion 特征
#         z2 = torch.tanh(self.linear_attn(attn_output))    # 融合 audio 后特征
#         g = torch.sigmoid(self.gate(torch.cat([z1, z2], dim=-1)))
#         fused = g * z1 + (1 - g) * z2                     # 门控融合

#         return fused   # 融合后的 motion 特征

# class MyCrossAttentionBlock(nn.Module):
#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.,
#             proj_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             mlp_layer=Mlp,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
        
#         self.cross_attn = CrossAttnGatedFusion(
#             dim,
#             num_heads=num_heads,
#         )
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = mlp_layer(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#         )
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x, context):
#         x = x + self.drop_path1(self.ls1(
#             self.cross_attn(self.norm1(x), context)
#         ))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x

# class MultiHeadLocalSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, window_size):
#         super(MultiHeadLocalSelfAttention, self).__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.head_dim = embed_dim // num_heads

#         # Projection layers
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         """
#         x: [B, T, D]
#         """
#         B, T, D = x.shape
#         H = self.num_heads
#         d = self.head_dim

#         # Project input to multi-head Q, K, V
#         Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
#         K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
#         V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]

#         # Padding K and V along time dimension
#         pad = self.window_size
#         K_padded = F.pad(K, (0, 0, pad, pad), mode='constant', value=0)  # [B, H, T+2w, d]
#         V_padded = F.pad(V, (0, 0, pad, pad), mode='constant', value=0)

#         outputs = []

#         for t in range(T):
#             # Extract local window for each time step
#             K_local = K_padded[:, :, t:t + 2 * pad + 1, :]  # [B, H, 2w+1, d]
#             V_local = V_padded[:, :, t:t + 2 * pad + 1, :]  # [B, H, 2w+1, d]
#             Q_t = Q[:, :, t:t + 1, :]  # [B, H, 1, d]

#             # Scaled dot-product attention
#             attn_scores = torch.matmul(Q_t, K_local.transpose(-2, -1)) / (d ** 0.5)  # [B, H, 1, 2w+1]
#             attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, 2w+1]
#             out_t = torch.matmul(attn_weights, V_local)  # [B, H, 1, d]

#             outputs.append(out_t)

#         # Concatenate over time
#         out = torch.cat(outputs, dim=2)  # [B, H, T, d]
#         out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

#         return self.out_proj(out)  # [B, T, D]



class EfficientMultiHeadLocalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(EfficientMultiHeadLocalSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim
        W = self.window_size

        Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]

        # Padding: [B, H, T, d] -> [B*H, T, d]
        K = K.reshape(B * H, T, d).transpose(1, 2)  # [B*H, d, T]
        V = V.reshape(B * H, T, d).transpose(1, 2)  # [B*H, d, T]

        pad = W
        K_padded = F.pad(K, (pad, pad), mode='constant', value=0)  # [B*H, d, T+2W]
        V_padded = F.pad(V, (pad, pad), mode='constant', value=0)

        # unfold to get local windows: [B*H, d, T, 2W+1]
        K_local = K_padded.unfold(dimension=2, size=2*W+1, step=1)  # [B*H, d, T, 2W+1]
        V_local = V_padded.unfold(dimension=2, size=2*W+1, step=1)  # [B*H, d, T, 2W+1]

        # Reshape back: [B*H, T, d, 2W+1]
        K_local = K_local.permute(0, 2, 1, 3)
        V_local = V_local.permute(0, 2, 1, 3)

        # Query: [B, H, T, d] -> [B*H, T, d, 1]
        Q = Q.reshape(B * H, T, d).unsqueeze(-1)

        # Attention: [B*H, T, 1, 2W+1]
        attn_scores = torch.matmul(Q.transpose(2, 3), K_local) / (d ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum: [B*H, T, 1, d]
        out = torch.matmul(attn_weights, V_local.transpose(2, 3))  # [B*H, T, 1, d]
        out = out.squeeze(2)  # [B*H, T, d]

        # Reshape back: [B, H, T, d] -> [B, T, D]
        out = out.view(B, H, T, d).transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)  # [B, T, D]





class MyBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            window_size=4
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
   
        # self.attn = MultiHeadLocalSelfAttention(dim, num_heads, window_size=window_size)  # Example window size, adjust as neede
        self.attn = EfficientMultiHeadLocalSelfAttention(dim, num_heads, window_size=window_size)  # Example window size, adjust as neede
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
#################################################################################################################



class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context):
        x = x + self.drop_path1(self.ls1(
            self.cross_attn(self.norm1(x), context)
        ))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_first=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.temporal_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.spatial_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.spatial_first = spatial_first
        
        # RoPE embeddings for temporal and spatial dimensions
        self.temporal_pos = SinusoidalEmbeddings(self.head_dim)
        self.spatial_pos = SinusoidalEmbeddings(self.head_dim)
    
    def _apply_rope(self, x, pos_emb):
        # x shape: (batch_size * n, seq_len, dim) or (batch_size * seq_len, n_joints, dim)
        b, seq, d = x.shape
        x = x.view(b, seq, self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)  # (b, num_heads, seq, head_dim)
        x = x.reshape(b * self.num_heads, seq, -1)
        
        # Apply RoPE
        pos_emb = pos_emb(x)
        x, _ = apply_rotary_pos_emb(x, x, pos_emb)
        
        # Reshape back
        x = x.reshape(b, self.num_heads, seq, -1)
        x = x.permute(0, 2, 1, 3)  # (b, seq, num_heads, head_dim)
        x = x.reshape(b, seq, -1)
        return x
    
    def _apply_temporal_attention(self, x):
        b, n, seq_len, dim = x.shape
        temp_x = x.reshape(b * n, seq_len, dim)
        
        # Apply RoPE
        temp_x = self._apply_rope(temp_x, self.temporal_pos)
        
        # Apply attention
        temporal_out, _ = self.temporal_attention(temp_x, temp_x, temp_x)
        temporal_out = temporal_out + temp_x
        return temporal_out.reshape(b, n, seq_len, dim)
    
    def _apply_spatial_attention(self, x):
        b, n, seq_len, dim = x.shape
        spatial_x = x.permute(0, 2, 1, 3).reshape(b * seq_len, n, dim)
        
        # Apply RoPE
        spatial_x = self._apply_rope(spatial_x, self.spatial_pos)
        
        # Apply attention
        spatial_out, _ = self.spatial_attention(spatial_x, spatial_x, spatial_x)
        spatial_out = spatial_out + spatial_x
        return spatial_out.reshape(b, seq_len, n, dim).permute(0, 2, 1, 3)
        
    def forward(self, x):
        if self.spatial_first:
            x = self._apply_spatial_attention(x)
            x = self._apply_temporal_attention(x)
        else:
            x = self._apply_temporal_attention(x)
            x = self._apply_spatial_attention(x)
        return x


class SpatialTemporalBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
       
        self.spatial_temporal_attn = JointAttention(dim, num_heads=num_heads, dropout=attn_drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        bs, n_joints, seq_len, dim = x.shape
       
       # apply spatial, then temporal attention
        x = x + self.drop_path1(self.ls1(self.spatial_temporal_attn(self.norm2(x))))

        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x