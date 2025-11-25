import torch
import math
import torch.nn.functional as F
import warnings
import collections.abc as container_abcs
import torch.nn as nn
from einops import rearrange

# Extracted from vit_utils.py
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

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
            xt = xt.reshape(B, H*W, T, -1).permute(0,2,1,3).contiguous().reshape(B*T, H*W, -1)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = res_temporal.reshape(B, T, H*W, -1).permute(0,2,1,3).contiguous().reshape(B, T*H*W, -1)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = cls_token.reshape(B*T, 1, -1)
            xs = xt
            xs = xs.reshape(B, H*W, T, -1).permute(0,2,1,3).contiguous().reshape(B*T, H*W, -1)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = cls_token.reshape(B, T, -1)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = res_spatial.reshape(B, T, H*W, -1).permute(0,2,1,3).contiguous().reshape(B, T*H*W, -1)
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, with_qkv=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SupressedHead(nn.Module):
    def __init__(self, 
                 encoder_embed_dim=256, 
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1, 
                 patch_size=8, 
                 in_chans=3, 
                 img_shape=(256, 320), 
                 T=5, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_shape = img_shape
        self.T = T
        self.num_features_levels = 4  # From intermediate_layer_idx [2,5,8,11]
        
        Hp = img_shape[0] // patch_size
        Wp = img_shape[1] // patch_size
        self.num_patches_per_frame = Hp * Wp
        
        # Projection layers for each encoder feature level to decoder dim // num_levels for concat
        self.level_proj = nn.ModuleList([
            nn.Linear(encoder_embed_dim, decoder_embed_dim // self.num_features_levels) for _ in range(self.num_features_levels)
        ])
        
        # Positional embeddings (spatial, shared across time and levels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, decoder_embed_dim))  # +1 for cls
        
        # Time embeddings for decoder
        self.time_embed = nn.Parameter(torch.zeros(1, T, decoder_embed_dim))
        
        # Decoder blocks: MAE-like Transformer decoder layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            BasicBlock(
                dim=decoder_embed_dim, 
                num_heads=decoder_num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer
            ) for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Prediction head: linear to pixel values per patch
        self.pred_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_uniform_(m)
    
    def forward(self, features):
        B = features[0][1].shape[0]  # From cls_tokens shape (B, embed_dim)
        Hp, Wp = self.img_shape[0] // self.patch_size, self.img_shape[1] // self.patch_size
        N = Hp * Wp
        
        # Process each level
        level_patches = []
        cls_tokens_list = []
        for i, (patches, cls_tokens) in enumerate(features):
            # patches: (B*T, N, encoder_embed_dim)
            # cls_tokens: (B, encoder_embed_dim)
            proj_patches = self.level_proj[i](patches)  # (B*T, N, decoder_dim // 4)
            proj_cls = self.level_proj[i](cls_tokens)  # (B, decoder_dim // 4)
            level_patches.append(proj_patches)
            cls_tokens_list.append(proj_cls)
        
        # Fusion: concat along dim for patches and cls
        fused_patches = torch.cat(level_patches, dim=-1)  # (B*T, N, decoder_dim)
        fused_cls = torch.cat(cls_tokens_list, dim=-1)  # (B, decoder_dim)
        
        # Reshape patches to (B, T, N, decoder_dim)
        x = fused_patches.view(B, self.T, N, -1)
        
        # Add time embeddings
        time_emb = self.time_embed.unsqueeze(2).repeat(1, 1, N, 1)
        time_emb = time_emb.repeat(B, 1, 1, 1)
        x = x + time_emb
        
        # Add positional embeddings
        pos_emb_patch = self.pos_embed[:, 1:1 + N, :]
        pos_emb = pos_emb_patch.unsqueeze(0).repeat(1, self.T, 1, 1)
        pos_emb = pos_emb.repeat(B, 1, 1, 1)
        x = x + pos_emb
        
        # Flatten to sequence: (B*T, N, decoder_dim)
        x = x.view(B * self.T, N, -1)
        
        # Handle CLS: (B, decoder_dim) -> (B, T, decoder_dim) -> (B*T, 1, decoder_dim)
        fused_cls = fused_cls.unsqueeze(1).repeat(1, self.T, 1)
        fused_cls = fused_cls.view(B * self.T, 1, -1)
        
        # Concat CLS to patches: (B*T, 1 + N, decoder_dim)
        x = torch.cat([fused_cls, x], dim=1)
        
        # Pass through decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # Remove CLS: (B*T, N, decoder_dim)
        x = x[:, 1:, :]
        
        # Predict pixels: (B*T, N, p**2 * C)
        x = self.pred_head(x)
        
        # Unpatchify
        p = self.patch_size
        C = self.in_chans
        x = x.view(x.shape[0], Hp, Wp, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(x.shape[0], C, Hp * p, Wp * p)
        
        # Reshape to (B, T, C, H, W)
        x = x.view(B, self.T, C, self.img_shape[0], self.img_shape[1])
        
        return x

# Adapted module for reflection suppression
class AdaptedSpatioTemporalReflectionModule(nn.Module):
    def __init__(self, 
                in_channels=3, 
                embed_dim=256, 
                num_heads=4, 
                depth=2, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1, 
                patch_size=8, 
                img_shape=(256, 320),
                T=5,
                attention_type='divided_space_time',
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.attention_type = attention_type
        # self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed = PatchEmbed(img_size=img_shape, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # Positional embeddings from TimeSformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for cls
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Time embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, T, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)
        
        # Attention blocks with divided_space_time
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type='divided_space_time')
            for i in range(depth)])
        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1
        
        self.norm = norm_layer(embed_dim)
        self.supresse_head = SupressedHead(embed_dim, 2 * embed_dim, patch_size=patch_size, img_shape=img_shape, T=T)
        self.intermediate_layer_idx = [2, 5, 8, 11]
        self.feature_norm = True


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, A_seq, M_seq):
        B, T, C, H, W = A_seq.shape
        masked_input = A_seq * M_seq.expand_as(A_seq)
        
        # Patch embedding (from TimeSformer, per frame)
        x, Hp, Wp = self.patch_embed(masked_input)
        
        # Add cls token
        cls_tokens = torch.zeros(B*T, 1, self.embed_dim, device=x.device)
        x = torch.cat((cls_tokens, x), dim=1)  # [(B*T), 1+N, embed_dim]
        
        # Add positional embed (resized if needed)
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2).unsqueeze(0), size=(1, x.size(1)), mode='bilinear').squeeze(0).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed.expand(B*T, -1, -1)
        x = self.pos_drop(x)
        
        # Time embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        # cls time embed
        # cls_tokens = cls_tokens.view(B, 1, T, self.embed_dim).permute(0,1,3,2).contiguous().reshape(B * 1, T, self.embed_dim)
        # cls_tokens = cls_tokens + time_embed.expand(B, -1, -1)
        # cls_tokens = self.time_drop(cls_tokens)
        # cls_tokens = cls_tokens.reshape(B, T, self.embed_dim).mean(1).unsqueeze(1).contiguous()
        # x = torch.cat((cls_tokens, x), dim=1).contiguous()
        x = torch.cat((cls_tokens, x), dim=1)
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, B, T, Wp)  # Pass B, T, Wp for view in Block
            if i in self.intermediate_layer_idx:
                features.append(x)
        
        x = self.norm(x)
        if self.feature_norm:
            features = [self.norm(f) for f in features]
            class_tokens = [f[:,0] for f in features]
            features = [rearrange(f[:,1:], 'b (h w t) m -> (b t) (h w) m',b=B,t=T,h=Hp, w=Wp) for f in features]
            features = tuple(zip(features, class_tokens))
        
        output = self.supresse_head(features)

        return output

# Test with print for debug
A_seq = torch.randn(1, 5, 3, 256, 320)  # Small size to avoid tool memory issue
M_seq = torch.randn(1, 5, 1, 256, 320)
model = AdaptedSpatioTemporalReflectionModule(patch_size=16, embed_dim=128, depth=12)  # Adjust patch for small H/W
print("Input A_seq shape:", A_seq.shape)
out = model(A_seq, M_seq)
print("Output shape:", out.shape)