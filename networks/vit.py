import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from .vit_utils import *
from .vit_blocks import *


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
            x = blk(x, B, self.T, N)
        
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
        x = torch.sigmoid(x)
        return x

# Adapted module for reflection suppression
class SpatioTemporalReflectionModule(nn.Module):
    def __init__(self, 
                in_channels=3, 
                embed_dim=256, 
                num_heads=4, 
                depth=12, 
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
                norm_layer=nn.LayerNorm,
                feature_norm=True):
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
        self.sigmoid = nn.Sigmoid()
        
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
        
        # Output projection to restore original channels
        self.proj_out = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

        self.intermediate_layer_idx = [2, 5, 8, 11]
        self.sup_head = SupressedHead(
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=embed_dim * 8,
            decoder_depth=8,
            decoder_num_heads=8,
            img_shape=img_shape,
            T = T,
            patch_size=patch_size
        )
        self.feature_norm = feature_norm
        self.temporal_attn_hooks = []
        self.temporal_attention_maps = []
        self._register_temporal_hooks()
        self.features = None

    def _register_temporal_hooks(self):
        # 只注册一次
        if len(self.temporal_attn_hooks) > 0:
            return

        self.temporal_attention_maps.clear()

        def hook_fn(module, input, output):
            if hasattr(module, 'attn') and module.attn is not None:
                # 用clone() 复制，保持梯度流
                self.temporal_attention_maps.append(module.attn.clone())

        for blk in self.blocks:
            if hasattr(blk, 'temporal_attn'):
                hook = blk.temporal_attn.register_forward_hook(hook_fn)
                self.temporal_attn_hooks.append(hook)

    def remove_temporal_hooks(self):
        for hook in self.temporal_attn_hooks:
            hook.remove()
        self.temporal_attn_hooks.clear()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, A_seq, M_seq):
        # 每个forward只清空maps
        self.temporal_attention_maps.clear()

        B, T, C, H, W = A_seq.shape
        masked_input = A_seq
        x, Hp, Wp = self.patch_embed(masked_input)
        cls_tokens = torch.zeros(B*T, 1, self.embed_dim, device=x.device)
        x = torch.cat((cls_tokens, x), dim=1)  
        
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2).unsqueeze(0), size=(1, x.size(1)), mode='bilinear').squeeze(0).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed.expand(B*T, -1, -1)
        x = self.pos_drop(x)
        
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
        x = torch.cat((cls_tokens, x), dim=1)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, B, T, Wp)  
            if i in self.intermediate_layer_idx:
                features.append(x)
        
        x = self.norm(x)
        if self.feature_norm:
            features = [self.norm(f) for f in features]
            class_tokens = [f[:,0] for f in features]
            features = [rearrange(f[:,1:], 'b (h w t) m -> (b t) (h w) m',b=B,t=T,h=Hp, w=Wp) for f in features]
            features = tuple(zip(features, class_tokens))
        suppress = self.sup_head(features)
        self.features = features    
        suppress = F.relu(suppress)
        return suppress


class SpatioTemporalDeformableReflectionModule(nn.Module):
    def __init__(self, 
                in_channels=3, 
                embed_dim=256, 
                num_heads=4, 
                depth=12, 
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
                norm_layer=nn.LayerNorm,
                feature_norm=True):
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
        self.sigmoid = nn.Sigmoid()
        
        # Time embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, T, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)
        
        # Attention blocks with divided_space_time
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DeformableBlock(
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
        
        # Output projection to restore original channels
        self.proj_out = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

        self.intermediate_layer_idx = [2, 5, 8, 11]
        self.sup_head = SupressedHead(
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=embed_dim * 8,
            decoder_depth=8,
            decoder_num_heads=8,
            img_shape=img_shape,
            T = T,
            patch_size=patch_size
        )
        self.feature_norm = feature_norm
        self.temporal_attn_hooks = []
        self.temporal_attention_maps = []
        self._register_temporal_hooks()
        self.features = None

    def _register_temporal_hooks(self):
        # 只注册一次
        if len(self.temporal_attn_hooks) > 0:
            return

        self.temporal_attention_maps.clear()

        def hook_fn(module, input, output):
            if hasattr(module, 'attn') and module.attn is not None:
                # 用clone() 复制，保持梯度流
                self.temporal_attention_maps.append(module.attn.clone())

        for blk in self.blocks:
            if hasattr(blk, 'temporal_attn'):
                hook = blk.temporal_attn.register_forward_hook(hook_fn)
                self.temporal_attn_hooks.append(hook)

    def remove_temporal_hooks(self):
        for hook in self.temporal_attn_hooks:
            hook.remove()
        self.temporal_attn_hooks.clear()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, A_seq, M_seq):
        # 每个forward只清空maps
        self.temporal_attention_maps.clear()

        B, T, C, H, W = A_seq.shape
        masked_input = A_seq
        x, Hp, Wp = self.patch_embed(masked_input)
        cls_tokens = torch.zeros(B*T, 1, self.embed_dim, device=x.device)
        x = torch.cat((cls_tokens, x), dim=1)  
        
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2).unsqueeze(0), size=(1, x.size(1)), mode='bilinear').squeeze(0).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed.expand(B*T, -1, -1)
        x = self.pos_drop(x)
        
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
        x = torch.cat((cls_tokens, x), dim=1)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, B, T, Wp)  
            if i in self.intermediate_layer_idx:
                features.append(x)
        
        x = self.norm(x)
        if self.feature_norm:
            features = [self.norm(f) for f in features]
            class_tokens = [f[:,0] for f in features]
            features = [rearrange(f[:,1:], 'b (h w t) m -> (b t) (h w) m',b=B,t=T,h=Hp, w=Wp) for f in features]
            features = tuple(zip(features, class_tokens))
        suppress = self.sup_head(features)
        self.features = features    
        suppress = F.relu(suppress)
        return suppress