import torch


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