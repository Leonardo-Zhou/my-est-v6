import torch
import matplotlib.pyplot as plt
import networks
from visualize_utils import *
import utils
from depth_anything import dpt
import cv2
import torch
import layers
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

frame_indices = [0, -1, 1]

def pose_predict(models, images, outputs):
    pose_feats = {
        f_i: images[f_i + 1].unsqueeze(0) for f_i in frame_indices
    }
    for f_i in frame_indices[1:]:
        if f_i < 0:
            inputs_all = [pose_feats[f_i], pose_feats[0]]
        else:
            inputs_all = [pose_feats[0], pose_feats[f_i]]
    
        pose_inputs = [models["pose_encoder"](torch.cat(inputs_all, 1))]
        axisangle, translation = models["pose"](pose_inputs)

        outputs[("axisangle", f_i)] = axisangle
        outputs[("translation", f_i)] = translation
        outputs[("cam_T_cam", f_i)] = layers.transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

K = np.array([[0.82, 0, 0.5, 0],
            [0, 1.02, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
K[0, :] *= 320
K[1, :] *= 256
inv_K = np.linalg.pinv(K)
K = torch.from_numpy(K).cuda().unsqueeze(0)
inv_K = torch.from_numpy(inv_K).cuda().unsqueeze(0)

B = 1
T = 3
width = 320
height = 256
patch = 16
Hp = height // patch
Wp = width // patch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model = networks.DepthSepQKV(resize_shape=(height, width), pretrained_path="./checkpoints/Depth-Anything-V2-Small-hf").to(DEVICE)
image_loader = ImageLoader(6, 3)
image = image_loader.single(1).to(DEVICE)
depth = model(image)
norm_cal = layers.NormalCalculator(height, width, 1).to(DEVICE)
norm = norm_cal.forward_from_depth(depth, K)