import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math



class DynamicReflectionModule(nn.Module):
    def __init__(self, in_channels=3, dim=256, heads=4, layers=2, dropout=0.1):
        """
        Dynamic Reflection Module (DRM)
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            dim (int, optional): Dimension of the transformer. Defaults to 256.
            heads (int, optional): Number of heads in the transformer. Defaults to 4.
            layers (int, optional): Number of layers in the transformer. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.

        """
        super().__init__()
        self.proj_in = nn.Conv3d(in_channels, dim, kernel_size=1)
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj_out = nn.Conv3d(dim, in_channels, kernel_size=1)

    def forward(self, A_seq, M_seq):
        """
        Dynamic Reflection Module (DRM)
        Args:
            A_seq (torch.Tensor): Input sequence of shape (B, T, C, H, W).
            M_seq (torch.Tensor): Mask sequence of shape (B, T, 1, H, W).

        Returns:
            torch.Tensor: Suppressed sequence of shape (B, T, C, H, W).
        """
        feat = self.proj_in((A_seq * M_seq.expand_as(A_seq)).permute(0, 2, 1, 3, 4))
        B, C, T, H, W = feat.shape
        feat_seq = feat.flatten(2).permute(0, 2, 1)
        attn_out = self.transformer(feat_seq)
        suppressed_seq = attn_out.permute(0, 2, 1).view(B, C, T, H, W)
        return self.proj_out(suppressed_seq).permute(0,2,1,3,4)




class ViTReflectionModule(nn.Module):
    def __init__(self, in_channels=3, dim=256, heads=4, layers=2, dropout=0.1, patch_size=16):
        """
        Vision Transformer based Reflection Module
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            dim (int, optional): Dimension of the transformer. Defaults to 256.
            heads (int, optional): Number of heads in the transformer. Defaults to 4.
            layers (int, optional): Number of layers in the transformer. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            patch_size (int, optional): Size of patches. Defaults to 16.
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Output projection
        self.proj_out = nn.ConvTranspose2d(dim, in_channels, kernel_size=patch_size, stride=patch_size)  # 用ConvTranspose上采，避免bilinear伪影
        
    def forward(self, A_seq, M_seq):
        """
        ViT Reflection Module
        Args:
            A_seq (torch.Tensor): Input sequence of shape (B, T, C, H, W).
            M_seq (torch.Tensor): Mask sequence of shape (B, T, 1, H, W).

        Returns:
            torch.Tensor: Suppressed sequence of shape (B, T, C, H, W).
        """
        B, T, C, H, W = A_seq.shape
        masked_input = A_seq * M_seq.expand_as(A_seq).view(B*T, C, H, W)  # 批量展平序列
        
        # Patch embedding
        patches = self.patch_embed(masked_input)  # [B*T, dim, Hp, Wp]
        
        # Add dynamic 2D spatial PE
        Hp, Wp = patches.shape[-2:]
        pos_h = torch.arange(Hp, device=patches.device).unsqueeze(1).repeat(1, Wp)
        pos_w = torch.arange(Wp, device=patches.device).unsqueeze(0).repeat(Hp, 1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=patches.device) * -(math.log(10000.0) / self.dim))
        spatial_pe = torch.zeros(Hp, Wp, self.dim, device=patches.device)
        spatial_pe[:, :, 0::2] = torch.sin(pos_h.unsqueeze(2) * div_term) + torch.sin(pos_w.unsqueeze(2) * div_term)
        spatial_pe[:, :, 1::2] = torch.cos(pos_h.unsqueeze(2) * div_term) + torch.cos(pos_w.unsqueeze(2) * div_term)
        patches = patches + spatial_pe.permute(2,0,1).unsqueeze(0).expand(B*T, -1, -1, -1)  # 空间PE
        
        # Add dynamic time PE (to capture sequence info in batch)
        time_pe = torch.zeros(B*T, self.dim, 1, 1, device=patches.device)
        pos_t = torch.arange(T, device=patches.device).unsqueeze(0).repeat(B, 1).view(B*T)
        time_pe[:, 0::2, :, :] = torch.sin(pos_t.unsqueeze(1) * div_term[:self.dim//2]).unsqueeze(-1).unsqueeze(-1)
        time_pe[:, 1::2, :, :] = torch.cos(pos_t.unsqueeze(1) * div_term[:self.dim//2]).unsqueeze(-1).unsqueeze(-1)
        patches = patches + time_pe.expand(-1, -1, Hp, Wp)  # 时间PE广播
        
        # Reshape for transformer: [B*T, N, dim]
        patches = patches.flatten(2).transpose(1, 2)  # [B*T, N, dim]
        
        # Apply transformer
        transformed = self.transformer(patches)  # [B*T, N, dim]
        
        # Reshape back
        transformed = transformed.transpose(1, 2).view(B*T, self.dim, Hp, Wp)  # [B*T, dim, Hp, Wp]
        
        # Project and upsample (ConvTranspose直接到原大小)
        output = self.proj_out(transformed)  # [B*T, C, H, W] 直接上采
        
        return output.view(B, T, C, H, W)







class SpatioTemporalReflectionModule(nn.Module):
    def __init__(self, in_channels=3, dim=256, heads=4, layers=2, dropout=0.1, patch_size=8, time_stride=1):
        """
        时空Transformer基于的反射模块
        Args:
            in_channels (int, optional): 输入通道数. 默认3 (RGB).
            dim (int, optional): Transformer维度. 默认256.
            heads (int, optional): 多头注意力头数. 默认4.
            layers (int, optional): Transformer层数. 默认2.
            dropout (float, optional): Dropout率. 默认0.1.
            patch_size (int, optional): 空间补丁大小. 默认8.
            time_stride (int, optional): 时间维度步长，用于压缩T以减复杂度. 默认1.
        """
        super().__init__()
        self.patch_size = patch_size  # 空间补丁大小，用于将图像分成小块，减序列长N = T* (H/patch)*(W/patch)，降低注意力O(N^2)复杂度
        self.time_stride = time_stride  # 时间步长，压缩时间维，减N大小，优化内存/速度（论文短序列隐含）
        self.dim = dim  # Transformer隐藏维度，平衡表达力与计算

        self.div_coeff = math.log(10000.0)
        
        # 时空补丁嵌入：Conv3d将输入序列分成时空补丁，同时投影到dim维
        # kernel=(3,patch,patch): 时间kernel=3捕捉相邻帧动态（反射移动），空间patch压缩
        # stride=(time_stride,patch,patch): 时间压缩减Tp = floor((T -3)/time_stride) +1
        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=(3, patch_size, patch_size), 
                                     stride=(time_stride, patch_size, patch_size))
        
        # 时空Transformer编码器：多层自注意力，捕捉时空依赖（反射跨帧/空间模式）
        encoder_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=layers)
        
        # 输出投影与上采样：ConvTranspose3d反卷积恢复原大小，kernel/stride匹配embed，避免bilinear伪影（论文梯度需尖锐）
        self.proj_out = nn.ConvTranspose3d(dim, in_channels, kernel_size=(3, patch_size, patch_size), 
                                           stride=(time_stride, patch_size, patch_size))
        
    def _get_3d_pe(self, T_p, H_p, W_p, dim, device):
        """
        数学公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))  # 偶数维
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  # 奇数维
        这里pos分T,H,W三个维度additive相加，捕捉多维位置（时间pos_t for动态反射，空间pos_h/w for纹理）。
        div_term = exp(arange(0,dim,2) * -log(10000)/dim) = 1 / 10000^(2i/dim)，length=dim//2=128 (for dim=256)，匹配pe[...,0::2] [T_p,H_p,W_p,dim//2]。
        为什么这样：不同频率（高i低波长）编码不同尺度位置，sin/cos周期性鲁棒。3D扩展处理内镜序列反射跨帧变化（论文页2 Fig.1光斑时间变异）。
        依据：原始PE论文；时空扩展VideoMAE (Tong et al., CVPR 2022)或TimeSformer，用于医疗视频捕捉动态（如反射抑制不丢纹理，论文Eq.9梯度软加权）。
        """
        pe = torch.zeros(T_p, H_p, W_p, dim, device=device)  # 初始化PE张量 [Tp, Hp, Wp, dim]
        pos_t = torch.arange(T_p, device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 时间位置 [Tp,1,1,1]
        pos_h = torch.arange(H_p, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 高度位置 [1,Hp,1,1]
        pos_w = torch.arange(W_p, device=device).unsqueeze(0).unsqueeze(1).unsqueeze(3)  # 宽度位置 [1,1,Wp,1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(self.div_coeff / dim))  # div_term [dim//2=128]，标准公式，确保频率渐减（高i低波长捕精细位置）
        pe[..., 0::2] = torch.sin(pos_t * div_term) + torch.sin(pos_h * div_term) + torch.sin(pos_w * div_term)  # 偶数维：3D相加sin，广播到 [Tp,Hp,Wp,dim//2]
        pe[..., 1::2] = torch.cos(pos_t * div_term) + torch.cos(pos_h * div_term) + torch.cos(pos_w * div_term)  # 奇数维：cos相加
        return pe.flatten(0,2).unsqueeze(0).expand(-1, -1, dim).transpose(1,2)  # [1, dim, Tp*Hp*Wp]，用于add到patches.flatten(2)

    def forward(self, A_seq, M_seq):
        """
        时空反射模块前向传播
        Args:
            A_seq (torch.Tensor): 输入序列 [B, T, C, H, W] (albedo序列，论文IID输出).
            M_seq (torch.Tensor): 掩码序列 [B, T, 1, H, W] (软/硬掩码，抑制反射区域).

        Returns:
            torch.Tensor: 抑制序列 [B, T, C, H, W] (反射淡化特征，保留纹理).
        """
        # 应用掩码：软弱化反射区域（论文Eq.9加权，保留上下文）
        masked_input = A_seq * M_seq.expand_as(A_seq)
        
        # 转置为Conv3d输入 [B, C, T, H, W]
        x = masked_input.permute(0, 2, 1, 3, 4)
        
        # 补丁嵌入：Conv3d投影/压缩到dim维补丁 [B, dim, Tp, Hp, Wp]
        patches = self.patch_embed(x)  # Tp = floor((T -3)/time_stride) +1, 捕捉局部时空
        
        # 添加动态3D PE：注入位置信息（包含时间pos_t，捕捉序列动态）
        B_p, D_p, T_p, H_p, W_p = patches.shape
        pe = self._get_3d_pe(T_p, H_p, W_p, D_p, patches.device)
        patches = patches + pe.unsqueeze(0).expand(B_p, -1, -1, -1, -1).view(B_p, D_p, T_p, H_p, W_p)  # broadcast add
        
        # 重塑为Transformer序列 [B, N, dim], N=Tp*Hp*Wp (时空展平，注意力全局捕捉)
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, dim]
        
        # 应用Transformer：自注意力学习抑制（反射模式权重低，纹理高）
        transformed = self.transformer(patches)  # [B, N, dim]
        
        # 重塑回3D [B, dim, Tp, Hp, Wp]
        transformed = transformed.transpose(1, 2).view(B_p, D_p, T_p, H_p, W_p)
        
        # 输出投影/上采样：ConvTranspose恢复原大小 [B, C, T, H, W]
        output = self.proj_out(transformed)
        
        return output.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]


if __name__=="__main__":
    # Test original module
    dynamic_reflection_module = DynamicReflectionModule().to(torch.device("cuda"))
    A_seq = torch.randn(1, 5, 3, 256, 320).to(torch.device("cuda"))
    M_seq = torch.randn(1, 5, 1, 256, 320).to(torch.device("cuda"))
    # out = dynamic_reflection_module(A_seq, M_seq)
    # print("Original DRM output shape:", out.shape)
    
    # Test ViT module
    # vit_reflection_module = ViTReflectionModule().to(torch.device("cuda"))
    # out_vit = vit_reflection_module(A_seq, M_seq)
    # print("ViT Reflection Module output shape:", out_vit.shape)
    
    # Test Spatio-Temporal module
    st_reflection_module = SpatioTemporalReflectionModule().to(torch.device("cuda"))
    out_st = st_reflection_module(A_seq, M_seq)
    print("Spatio-Temporal Reflection Module output shape:", out_st.shape)