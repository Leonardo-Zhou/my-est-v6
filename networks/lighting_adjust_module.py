import torch
import torch.nn as nn
import torch.nn.functional as F

class LAM(nn.Module):
    def __init__(self):
        super().__init__()
        # 示例：简单卷积网络，输入 2 通道 (D + I_gray)，输出 1 通道 LA
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, D, I_gray):
        input_cat = torch.cat([D, I_gray], dim=1)  # [B, 2, H, W]
        x = self.relu(self.conv1(input_cat))
        LA = self.conv2(x)  # [B, 1, H, W]
        return LA