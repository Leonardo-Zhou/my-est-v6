from depth_anything import dpt, blocks, transform, dinov2
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizeImageBatch(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std, device='cuda'):
        self.__mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, -1, 1, 1).to(device) 

    def __call__(self, sample):
        sample = (sample - self.__mean) / self.__std
        return sample



class Depth(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        resize_shape=(224, 280),
        device='cuda'
    ):
        super(Depth, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = dinov2.DINOv2(model_name=encoder)
        self.depth_head = dpt.DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.normalize = NormalizeImageBatch(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], device=device)
        self.resize_shape = resize_shape

    def forward(self, x):
        shape = x.shape[-2:]
        x = F.interpolate(x, self.resize_shape, mode="bilinear", align_corners=True)
        x = self.normalize(x)
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        depth = F.interpolate(depth, shape, mode="bilinear", align_corners=True)
        return depth