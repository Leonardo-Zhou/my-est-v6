import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

def enhance_brightness_torch(image_tensor: torch.Tensor, factor) -> torch.Tensor:
    """
    使用 PyTorch 快速增强图像亮度 (C, H, W) 或 (B, C, H, W)。
    
    假设：输入 image_tensor 的值范围在 [0.0, 1.0] 之间。

    参数:
        image_tensor (torch.Tensor): 输入的图像张量，形状为 (C, H, W) 或 (B, C, H, W)。
        factor (float or torch.Tensor): 亮度增强因子。
                        如果是float，> 1.0 会增加亮度，< 1.0 会降低亮度，= 1.0 保持不变。
                        如果是torch.Tensor，形状为(B,)，每个元素对应一个样本的增强因子。

    返回:
        torch.Tensor
    """
    # 处理负数因子
    if isinstance(factor, (int, float)) and factor < 0:
        factor = 0.0
    elif isinstance(factor, torch.Tensor):
        factor = torch.clamp(factor, min=0.0)
    
    # 如果factor是tensor且image_tensor是batch形式
    if isinstance(factor, torch.Tensor) and len(image_tensor.shape) == 4:
        # factor形状: (B,)
        # image_tensor形状: (B, C, H, W)
        B, C, H, W = image_tensor.shape
        
        # 将factor reshape为 (B, 1, 1, 1) 以便广播
        factor = factor.view(B, 1, 1, 1)
    
    enhanced_image = image_tensor * factor
    enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)
    
    return enhanced_image

class FactorChoicer:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.mul_factor = torch.tensor([0.2], device=device)
        self.add_factor = torch.tensor([0.0, 1.0, 0.8], device=device).reshape(3, 1)

    def get_factor(self, enhance_type=-1) -> torch.Tensor:
        """
        获取增强因子
        
        参数:
            enhance_type (int): 增强类型，-1为亮度减弱，1为亮度增强
        
        返回:
            torch.Tensor: 形状为(batch_size,)的tensor，每个元素是随机选择的增强因子
        """
        rands = torch.rand(3, self.batch_size, device=self.device)
        factors = rands * self.mul_factor + self.add_factor
        factors = factors[enhance_type]
        return factors

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


class NormalizeImageBatch(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std, device="cpu"):
        self.__mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def __call__(self, sample):
        sample = (sample - self.__mean) / self.__std

        return sample


def extract_samples(suppressed_seq, M_seq, num_pos=128, num_neg=128):
    B, T, C, H, W = suppressed_seq.shape
    flat_feat = suppressed_seq.view(B, T, C, -1)  # [B, T, C, HW]
    flat_M = M_seq.view(B, T, 1, -1)
    mid = T // 2  # 中间帧索引

    # 正样本：相邻帧 (t-1, t+1) 非反射平均 [B, C]
    non_reflect_mask = (flat_M[:, mid-1:mid+2] < 0.5).squeeze(2).float()  # [B, 3, HW]

    # 获取相邻帧的特征 [B, 2, C, HW]
    pos_feat = flat_feat[:, mid-1:mid+2]  # [B, 2, C, HW]
    pos_feat_flat = pos_feat.permute(0, 2, 1, 3).reshape(B, C, -1)  # [B, C, 3*HW]
    
    # 从相邻帧的非反射区域采样
    non_reflect_mask_adj = non_reflect_mask.reshape(B, -1)  # [B, 3*HW]
    
    # 检查是否有有效的非反射区域
    for b in range(B):
        if non_reflect_mask_adj[b].sum() == 0:
            # 如果没有非反射区域，使用所有像素
            non_reflect_mask_adj[b] = torch.ones_like(non_reflect_mask_adj[b])
    
    pos_indices = torch.multinomial(non_reflect_mask_adj, num_pos, replacement=True)
    
    # 采样并平均
    pos_samples = torch.gather(pos_feat_flat, 2, pos_indices.unsqueeze(1).expand(-1, C, -1))  # [B, C, num_pos]
    pos = pos_samples.mean(dim=2)  # [B, C]

    # 负样本：当前序列反射平均 [B, C]
    reflect_mask = (flat_M > 0.5).squeeze(2).float()  # [B, T, HW]
    reflect_mask_flat = reflect_mask.reshape(B, -1)  # [B, T*HW]
    
    # 检查是否有有效的反射区域
    for b in range(B):
        if reflect_mask_flat[b].sum() == 0:
            # 如果没有反射区域，使用所有像素
            reflect_mask_flat[b] = torch.ones_like(reflect_mask_flat[b])
    
    neg_indices = torch.multinomial(reflect_mask_flat, num_neg, replacement=True)
    
    # 获取所有时间步的特征 [B, T*C, HW]
    neg_feat_flat = flat_feat.permute(0, 2, 1, 3).reshape(B, C, -1)  # [B, C, T*HW]
    neg_samples = torch.gather(neg_feat_flat, 2, neg_indices.unsqueeze(1).expand(-1, C, -1))  # [B, C, num_neg]
    neg = neg_samples.mean(dim=2)  # [B, C]
    
    return pos, neg

# 时空对比损失（同前，但时空anchor）
def contrastive_loss(anchor, pos, neg, tau=0.07):
    anchor, pos, neg = map(F.normalize, [anchor, pos, neg])
    sim_pos = F.cosine_similarity(anchor, pos, dim=-1)
    sim_neg = F.cosine_similarity(anchor, neg, dim=-1)
    logits = torch.stack([sim_pos, sim_neg], dim=1) / tau
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_smooth_loss(disp, img):
    
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def visualize_depth(depth):
    """
    depth: (H, W)
    """
    depth=depth.squeeze()
    x = depth.cpu().detach().numpy()
    vmax = np.percentile(x, 95)

    normalizer = mpl.colors.Normalize(vmin=x.min(), vmax=vmax) # 归一化到0-1
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
    colormapped_im = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im=np.transpose(colormapped_im,(2,0,1))
    return colormapped_im
