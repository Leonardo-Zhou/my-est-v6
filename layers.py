from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import math
import cv2
import torch.nn as nn
import torch.nn.functional as F

from warnings import warn


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


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords


class Project3D_Raw(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_Raw, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        raw_pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        raw_pix_coords = raw_pix_coords.view(self.batch_size, 2, self.height, self.width)
        raw_pix_coords = raw_pix_coords.permute(0, 2, 3, 1)

        return raw_pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img, mask=None):
    
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    if mask is not None:
        disp = disp * mask
        img = img * mask

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_smooth_light(light,img):
    grad_light_x = torch.abs(light[:, :, :, :-1] - light[:, :, :, 1:])
    grad_light_y = torch.abs(light[:, :, :-1, :] - light[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    epsilon_x = 0.01*torch.ones_like(grad_img_x)
    Denominator_x = torch.max(grad_img_x, epsilon_x)
    x_loss = torch.abs(torch.div(grad_light_x, Denominator_x))

    epsilon_y = 0.01*torch.ones_like(grad_img_y)
    Denominator_y = torch.max(grad_img_y, epsilon_y)
    y_loss = torch.abs(torch.div(grad_light_y, Denominator_y))
    
    return x_loss.mean() + y_loss.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    abs_diff=torch.mean(torch.abs(gt - pred))

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        """
        Instiantiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the source image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode="border",align_corners=True)

# EndoDAC 使用(v7)
class get_occu_mask_backward(nn.Module):

    def __init__(self, size):
        super(get_occu_mask_backward, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, flow, th=0.95):
        
        new_locs = self.grid + flow
        new_locs = new_locs[:, [1,0], ...]
        corr_map = get_corresponding_map(new_locs)
        occu_map = corr_map
        occu_mask = (occu_map > th).float()

        return occu_mask, occu_map
# EndoDAC 使用(v7)
class get_occu_mask_bidirection(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super(get_occu_mask_bidirection, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, flow12, flow21, scale=0.01, bias=0.5):
        
        new_locs = self.grid + flow12
        shape = flow12.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:, i, ...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        flow21_warped = F.grid_sample(flow21, new_locs, mode=self.mode, padding_mode="border")
        flow12_diff = torch.abs(flow12 + flow21_warped)
        # mag = (flow12 * flow12).sum(1, keepdim=True) + \
        # (flow21_warped * flow21_warped).sum(1, keepdim=True)
        # occ_thresh = scale * mag + bias
        # occ_mask = (flow12_diff * flow12_diff).sum(1, keepdim=True) < occ_thresh
        
        return flow12_diff
# EndoDAC 使用(v7)
class optical_flow(nn.Module):

    def __init__(self, size, batch_size, height, width, eps=1e-7):
        super(optical_flow, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        optical_flow =  pix_coords[:, [1,0], ...] - self.grid

        return optical_flow
# EndoDAC 使用(v7)
def get_smooth_bright(transform, target, pred, occu_mask):
    
    """Computes the smoothness loss for a appearance flow
    """
    grad_transform_x = torch.mean(torch.abs(transform[:, :, :, :-1] - transform[:, :, :, 1:]), 1, keepdim=True)
    grad_transform_y = torch.mean(torch.abs(transform[:, :, :-1, :] - transform[:, :, 1:, :]), 1, keepdim=True)
     
    residue = (target - pred)
    
    grad_residue_x = torch.mean(torch.abs(residue[:, :, :, :-1] - residue[:, :, :, 1:]), 1, keepdim=True)
    grad_residue_y = torch.mean(torch.abs(residue[:, :, :-1, :] - residue[:, :, 1:, :]), 1, keepdim=True)

    mask_x = occu_mask[:, :, :, :-1]
    mask_y = occu_mask[:, :, :-1, :]

    # grad_residue_x = grad_residue_x * mask_x / (mask_x.mean() + 1e-7)
    # grad_residue_y = grad_residue_y * mask_y / (mask_y.mean() + 1e-7)
    
    grad_transform_x *= torch.exp(-grad_residue_x)
    grad_transform_y *= torch.exp(-grad_residue_y)

    grad_transform_x *= mask_x
    grad_transform_y *= mask_y
    
    return (grad_transform_x.sum() / mask_x.sum() + grad_transform_y.sum() / mask_y.sum())


def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, target):
        
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        
        diff = pred - target
        abs_diff = diff.abs()
        c = 0.2 * abs_diff.max() 
        mask = (abs_diff <= c).float()
        l2_loss = (diff ** 2 + c ** 2) / (2 * c)

        loss = (mask * abs_diff + (1 - mask) * l2_loss).mean()

        return loss


class reduced_ransac(nn.Module):
    def __init__(self, check_num, dataset):
        super(reduced_ransac, self).__init__()
        self.check_num = check_num
        # self.thres = thres
        self.dataset = dataset

    def robust_rand_sample(self, match, mask, num, robust=True):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1)) # []
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:,:,rand_int]
        else:
            # If there is zero score in match, sample the non-zero matches.
            select_idxs = []
            if robust:
                num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
            for i in range(b):
                nonzero_idx = torch.nonzero(mask[i,0,:]) # [nonzero_num,1]
                rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
                select_idx = nonzero_idx[rand_int, :] # [num, 1]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0) # [b,num,1]
            select_match = torch.gather(match.transpose(1,2), index=select_idxs.repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, num]
        return select_match, num

    def top_ratio_sample(self, match, mask, ratio):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, total_num = match.shape[0], match.shape[-1]
        scores, indices = torch.topk(mask, int(ratio*total_num), dim=-1) # [B, 1, ratio*tnum]
        select_match = torch.gather(match.transpose(1,2), index=indices.squeeze(1).unsqueeze(-1).repeat(1,1,4), dim=1).transpose(1,2) # [b, 4, ratio*tnum]
        return select_match, scores

    def forward(self, match, mask, visualizer=None):
        # match: [B, 4, H, W] mask: [B, 1, H, W]
        b, h, w = match.shape[0], match.shape[2], match.shape[3]
        match = match.view([b, 4, -1]).contiguous()
        mask = mask.view([b, 1, -1]).contiguous()
        
        # Sample matches for RANSAC 8-point and best F selection
        top_ratio_match, top_ratio_mask = self.top_ratio_sample(match, mask, ratio=0.20) # [b, 4, ratio*H*W] 
        check_match, check_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask, num=self.check_num) # [b, 4, check_num]
        check_match = check_match.contiguous()

        cv_f = []
        for i in range(b):
            if self.dataset == 'nyuv2':
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_LMEDS, 0.99)
            else:
                f, m = cv2.findFundamentalMat(check_match[i,:2,:].transpose(0,1).detach().cpu().numpy(), check_match[i,2:,:].transpose(0,1).detach().cpu().numpy(), cv2.FM_RANSAC, 0.1, 0.99)
            cv_f.append(f)
        cv_f = np.stack(cv_f, axis=0)
        cv_f = torch.from_numpy(cv_f).float().to(match.get_device())
        
        return cv_f


class Nabla:
    def __init__(self, device):
        # 1. 一次性 Sobel 核（3×3，可分离，3 通道 group 卷积）
        self.sobel = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        sobel_x = self.sobel.view(1,1,3,3).repeat(C,1,1,1)  # 3 通道 group=3
        sobel_y = self.sobel.t().view(1,1,3,3).repeat(C,1,1,1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=C)
        gy = F.conv2d(x, sobel_y, padding=1, groups=C)
        # 添加数值稳定性保护，避免在梯度接近0时出现梯度爆炸
        gradient_magnitude = gx**2 + gy**2
        # 使用较小的epsilon值防止sqrt(0)导致的梯度爆炸
        return (gradient_magnitude + 1e-8).sqrt()

    def __call__(self, x):
        return self.forward(x)

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, pred, target):
        return torch.abs(pred - target).mean()


import torch
import torch.nn as nn
import torch.nn.functional as F
import layers

class NormalCalculator(nn.Module):
    """根据论文公式从3D点云计算法线和不确定性图的模块。
    
    忠于论文公式(6)和(7)的计算方式。
    使用张量操作和广播进行向量化计算，避免Python循环，以支持CUDA加速。
    边界使用'replicate' padding处理。
    假设输入的3D点已处于相机坐标系中，Z轴指向前方（正Z）。
    """

    def __init__(self, height, width, batch_size=1):
        super(NormalCalculator, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        
        # 假设有backproject模块，与用户代码一致
        self.backproject = layers.BackprojectDepth(batch_size, height, width)  # 请替换为实际导入
        
    def forward_from_depth(self, depth, K):
        """
        
        Args:
            depth: [B, 1, H, W] 深度图
            K: [B, 4, 4] 相机内参
            
        Returns:
            normals: [B, 3, H, W] 法线图（单位向量）
        """
        B, _, H, W = depth.shape
        
        # 获取3D点云
        inv_K = torch.inverse(K)
        points_3d = self.backproject(depth, inv_K)
        
        return self.forward_from_points(points_3d)
    
    def forward_from_points(self, points_3d):
        """
        从3D点云计算法线（忠于论文公式(6)）
        
        论文公式: N_p^b = 1 / |Ω| ∑_{pi,pj ∈ Ω} [ (P_pi - P_p) × (P_pj - P_p) / ||(P_pi - P_p) × (P_pj - P_p)||_2 ]
        但公式中1/|Ω|与sum over pi,pj不匹配（|Ω|=8，但sum over 8x8=64项）。
        根据上下文和Fig.3(a)，这是对所有pi,pj对（包括i!=j）的归一化叉积平均。
        为解决方向问题：对每个叉积后，翻转使Z>0，然后平均所有非对角项，最后归一化。
        总项数：8*7=56（排除i==j）。
        
        Args:
            points_3d: [B, 4, H*W] 3D点云（齐次坐标）
            
        Returns:
            normals: [B, 3, H, W] 法线图（单位向量）
        """
        B = points_3d.shape[0]
        H, W = self.height, self.width
        
        # 提取XYZ并reshape为[B, 3, H, W]
        xyz = points_3d[:, :3].view(B, 3, H, W)
        
        # 使用replicate padding处理边界
        pad_xyz = F.pad(xyz, (1, 1, 1, 1), mode='replicate')
        
        # 提取8个邻域点：[B, 3, H, W, 8]
        offsets = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)]
        neigh_list = []
        for dy, dx in offsets:
            neigh = pad_xyz[:, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            neigh_list.append(neigh)
        neigh = torch.stack(neigh_list, dim=4)  # [B, 3, H, W, 8]
        
        # 计算所有pi,pj对的向量：[B, 3, H, W, 8, 8]
        vec1 = neigh.unsqueeze(5) - xyz.unsqueeze(4).unsqueeze(5)  # 广播到 [B, 3, H, W, 8, 1] - [B, 3, H, W, 1, 1]
        vec2 = neigh.unsqueeze(4) - xyz.unsqueeze(4).unsqueeze(5)  # 广播到 [B, 3, H, W, 1, 8] - [B, 3, H, W, 1, 1]
        
        # 计算叉积：[B, 3, H, W, 8, 8]
        cross = torch.cross(vec1, vec2, dim=1)
        
        # 计算L2范数：[B, 1, H, W, 8, 8]
        norm = torch.norm(cross, p=2, dim=1, keepdim=True) + 1e-8
        
        # 归一化叉积：[B, 3, H, W, 8, 8]
        n = cross / norm
        
        # 如果Z分量<0，翻转方向：[B, 3, H, W, 8, 8]
        flip_mask = n[:, 2, :, :, :, :] < 0  # [B, H, W, 8, 8]
        n = torch.where(flip_mask.unsqueeze(1), -n, n)
        
        # 掩码排除对角（i==j）：对角叉积为0
        diag_mask = 1.0 - torch.eye(8, device=n.device)  # [8, 8]，对角0，其他1
        diag_mask = diag_mask.view(1, 1, 1, 1, 8, 8)  # [1, 1, 1, 1, 8, 8]
        
        # 应用掩码并求和：[B, 3, H, W]
        masked_n = n * diag_mask
        sum_n = masked_n.sum(dim=[4, 5])
        
        # 平均（除以非对角项数56）
        num_pairs = 8.0 * 7.0
        avg_n = sum_n / num_pairs
        
        # 最终归一化平均向量：[B, 3, H, W]
        avg_norm = torch.norm(avg_n, p=2, dim=1, keepdim=True) + 1e-8
        normals = avg_n / avg_norm
        
        return normals
    
    def compute_uncertainty_from_points(self, normals, points_3d):
        """
        从3D点云和法线计算不确定性图（忠于论文公式(7)）
        
        论文公式: U_p = 1 / |Ω| ∑_{pk ∈ Ω} || N_p^b · (P_pk - P_p) ||_2
        其中||scalar||_2 等价于 |scalar|（距离的绝对值）。
        |Ω|=8。
        
        Args:
            normals: [B, 3, H, W] 已计算的法线（来自forward_from_points）
            points_3d: [B, 4, H*W] 3D点云（齐次坐标）
            
        Returns:
            uncertainty: [B, 1, H, W] 不确定性图
        """
        B = points_3d.shape[0]
        H, W = self.height, self.width
        
        # 提取XYZ并reshape为[B, 3, H, W]
        xyz = points_3d[:, :3].view(B, 3, H, W)
        
        # 复用邻域提取（与forward_from_points一致）
        pad_xyz = F.pad(xyz, (1, 1, 1, 1), mode='replicate')
        offsets = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)]
        neigh_list = []
        for dy, dx in offsets:
            neigh = pad_xyz[:, :, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            neigh_list.append(neigh)
        neigh = torch.stack(neigh_list, dim=4)  # [B, 3, H, W, 8]
        
        # 计算向量：[B, 3, H, W, 8]
        vec_k = neigh - xyz.unsqueeze(4)
        
        # 计算点积：[B, H, W, 8]
        dots = torch.sum(normals.unsqueeze(4) * vec_k, dim=1)
        
        # 绝对值（距离）：[B, H, W, 8]
        abs_dots = torch.abs(dots)
        
        # 平均过8个邻域：[B, H, W, 1]
        uncertainty = abs_dots.mean(dim=3, keepdim=True)
        
        # 转置为[B, 1, H, W]
        return uncertainty.permute(0, 3, 1, 2)
    
    def compute_normal_map(self, depth, K):
        """
        计算法线贴图，适用于可视化（与用户代码一致）
        
        Args:
            depth: [B, 1, H, W] 深度图
            K: [B, 4, 4] 相机内参
            
        Returns:
            normal_map: [B, 3, H, W] 法线贴图（范围[0,1]）
        """
        normals = self.forward_from_depth(depth, K)
        # 将法线从[-1,1]映射到[0,1]用于可视化
        normal_map = (normals + 1.0) * 0.5
        return torch.clamp(normal_map, 0.0, 1.0)