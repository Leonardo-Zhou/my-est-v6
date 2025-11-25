import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from .vit_utils import *

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# Extracted from vit.py
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn = None  # 初始化

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn = attn  # 保存干净的 attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(256, 320), patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        # x = x.reshape(B*T, C, H, W)
        x = self.proj(x)
        W, H = x.size(-1), x.size(-2)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, N):  # N = num_patches_per_frame
        # x: (B*T, 1 + N, dim)
        num_spatial = 1 + N  # incl CLS

        # Temporal attn (per spatial position)
        xt = x
        xt = xt.reshape(B, T, num_spatial, -1).permute(0, 2, 1, 3).reshape(B * num_spatial, T, -1)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = res_temporal.reshape(B, num_spatial, T, -1).permute(0, 2, 1, 3).reshape(B * T, num_spatial, -1)
        res_temporal = self.temporal_fc(res_temporal)
        x = x + res_temporal

        # Spatial attn
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableTemporalAttention(nn.Module):
    """
    2025年终极版：专为内窥镜高光抑制设计的 Deformable Temporal Attention
    完美解决相机移动 + patch 错位问题
    """
    def __init__(self, dim, num_heads=8, num_frames=3, num_points=9, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_points = num_points
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        # 预测 offset 和 attention weight（每个 head 独立）
        self.offset_proj = nn.Linear(dim, num_heads * num_points * 2)
        self.attn_proj = nn.Linear(dim, num_heads * num_points)
        
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_proj.bias, 0.)
        nn.init.constant_(self.attn_proj.bias, 0.)
        # offset 初始化为 0（中心采样）
        nn.init.constant_(self.offset_proj.weight, 0.)
        nn.init.constant_(self.attn_proj.weight, 0.)

    def forward(self, x, Hp, Wp):
        """
        x: (B, N, T, D)  # N = Hp * Wp
        Hp, Wp: patch 网格尺寸
        """
        B, N, T, D = x.shape
        mid_idx = T // 2

        # 只用中间帧做 query
        query = x[:, :, mid_idx]  # (B, N, D)

        # 生成参考点 (N, 2) → 归一化到 [-1, 1]
        h_idx = torch.arange(Hp, device=x.device).float()
        w_idx = torch.arange(Wp, device=x.device).float()
        grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
        ref_points = torch.stack([grid_w, grid_h], dim=-1).flatten(0, 1)  # (N, 2)
        ref_points = ref_points / torch.tensor([Wp-1, Hp-1], device=x.device) * 2 - 1  # [-1,1]
        ref_points = ref_points.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # 预测 offset 和 attention
        offset = self.offset_proj(query)  # (B, N, heads*points*2)
        attn_weight = self.attn_proj(query)  # (B, N, heads*points)

        offset = offset.view(B, N, self.num_heads, self.num_points, 2)
        attn_weight = attn_weight.view(B, N, self.num_heads, self.num_points)
        attn_weight = attn_weight.softmax(-1)  # 在采样点上 softmax

        # 计算采样位置
        sampling_locations = ref_points.unsqueeze(2).unsqueeze(2)  # (B, N, 1, 1, 2)
        sampling_locations = sampling_locations + offset  # (B, N, heads, points, 2)

        # value 从所有帧采样
        value = self.value_proj(x)  # (B, N, T, D)
        value = value.view(B, N, T, self.num_heads, self.dim_head)
        value = value.permute(0, 3, 1, 4, 2).contiguous()  # (B, heads, N, dim_head, T)

        # grid_sample 需要 (B*heads, dim_head, Hp, Wp, T) → 我们逐帧采样
        output = 0.0
        for t in range(T):
            # (B, heads, N, dim_head)
            value_t = value[..., t].contiguous()
            value_t = value_t.view(B * self.num_heads, N, self.dim_head)
            value_t = value_t.permute(0, 2, 1).contiguous()  # (B*heads, dim_head, N)
            value_t = value_t.view(B * self.num_heads, self.dim_head, Hp, Wp)

            # 采样（对每一帧）
            sampled = F.grid_sample(
                value_t,
                sampling_locations[:, :, t:t+1],  # 只采样第 t 帧
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B*heads, dim_head, N, points)

            sampled = sampled.view(B, self.num_heads, self.dim_head, N, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # (B, N, heads, points, dim_head)

            output += (sampled * attn_weight.unsqueeze(-1)).sum(3)  # 加权求和

        output = output.flatten(2)  # (B, N, D)
        output = self.out_proj(output)
        return output


class DeformableBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = DeformableTemporalAttention(
           dim, num_heads=num_heads, num_frames=3, num_points=9)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x