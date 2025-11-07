import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class AdjustNet(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super(AdjustNet, self).__init__()
        self.input_channel = num_input_channels
        self.output_channel = num_output_channels
        
        # For single input case
        self.convs = OrderedDict()
        self.convs[("conv", 1)] = ConvBlock(self.input_channel, 32)
        self.convs[("conv", 2)] = ConvBlock(32, 32)
        self.convs[("conv", 3)] = ConvBlock(32, 32)
        self.convs[("conv", 4)] = nn.Conv2d(32, self.output_channel, kernel_size=1)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.Tanh = nn.Tanh()
        
        # For multiple input case (A, S, R)
        # Each component gets its own adjustment network
        self.adjust_A = self._make_adjustment_net()
        self.adjust_S = self._make_adjustment_net()
        self.adjust_R = self._make_adjustment_net()
    
    def _make_adjustment_net(self):
        """Create a small adjustment network for a single component"""
        convs = OrderedDict()
        convs[("conv", 1)] = ConvBlock(self.input_channel, 32)
        convs[("conv", 2)] = ConvBlock(32, 32)
        convs[("conv", 3)] = ConvBlock(32, 32)
        convs[("conv", 4)] = nn.Conv2d(32, self.output_channel, kernel_size=1)
        return nn.Sequential(*list(convs.values()))
    
    def forward_single(self, input_features):
        """Process a single input feature map"""
        adjust_L = self.convs[("conv", 1)](input_features)
        adjust_L = self.convs[("conv", 2)](adjust_L)
        adjust_L = self.convs[("conv", 3)](adjust_L)
        adjust_L = self.convs[("conv", 4)](adjust_L)
        outputs = self.Tanh(adjust_L)
        return outputs
    
    def forward_multi(self, input_A, input_S, input_R):
        """Process three input feature maps (A, S, R)"""
        adjust_A = self.adjust_A(input_A)
        adjust_S = self.adjust_S(input_S)
        adjust_R = self.adjust_R(input_R)
        
        # Apply tanh activation to each output
        adjust_A = self.Tanh(adjust_A)
        adjust_S = self.Tanh(adjust_S)
        adjust_R = self.Tanh(adjust_R)
        
        return adjust_A, adjust_S, adjust_R
    
    def forward(self, *inputs):
        """Forward pass that handles both single and multiple inputs"""
        if len(inputs) == 1:
            # Single input case
            return self.forward_single(inputs[0])
        elif len(inputs) == 3:
            # Three inputs case (A, S, R)
            return self.forward_multi(inputs[0], inputs[1], inputs[2])
        else:
            raise ValueError(f"adjust_net expects 1 or 3 inputs, but got {len(inputs)}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Test single input
    model = adjust_net().cuda()
    model.eval()
    
    tgt_img = torch.randn(4, 3, 256, 320).cuda()
    output = model(tgt_img)
    print("Single input output shape:", output.shape)
    
    # Test three inputs
    input_A = torch.randn(4, 3, 256, 320).cuda()
    input_S = torch.randn(4, 3, 256, 320).cuda()
    input_R = torch.randn(4, 3, 256, 320).cuda()
    output_A, output_S, output_R = model(input_A, input_S, input_R)
    print("Multi input output shapes:", output_A.shape, output_S.shape, output_R.shape)
