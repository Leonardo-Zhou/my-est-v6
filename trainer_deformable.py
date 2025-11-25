from depth_anything import dpt
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import datasets
import networks
import torch.optim as optim
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import utils
import random
import os
from peft import LoraConfig, get_peft_model
from einops import rearrange



class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.frames = self.opt.frame_ids.copy()
        self.frames.sort()

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.model_init()
        print("Training Non-Lambertian model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        self.data_init()
        self.util_init()
        self.save_opts()

    def model_init(self):

        self.normal_lr_parameters = []
        self.low_lr_parameters = []

        if not self.opt.sep_qkv:
            # 初始化LoRA微调参数
            lora_config = LoraConfig(
                r=16,  # 低秩维度，平衡效率和性能
                lora_alpha=20,  # 缩放因子，通常为 2*r
                target_modules=["qkv"],  # 针对 ViT 层
                lora_dropout=0.05,  # dropout 防止过拟合
                bias="none",  # 不调整偏置
                task_type=None  # 自定义任务类型
            )

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            encoder = self.opt.vit_encoder

            model = networks.Depth(**model_configs[encoder], device=self.device)
            # 说明初始训练，从文件夹加载预训练权重
            if not self.opt.load_weights_folder or "depth" not in self.opt.models_to_load:
                model.load_state_dict(torch.load(os.path.join(self.opt.vit_folder, f'depth_anything_v2_{encoder}.pth'), map_location='cuda'))
        else:
            model = networks.DepthSepQKV(resize_shape=(self.opt.height, self.opt.width), pretrained_path=self.opt.da_sep_qkv_folder)
            # 初始化LoRA微调参数
            lora_config = LoraConfig(
                r=16,  # 低秩维度，平衡效率和性能
                lora_alpha=20,  # 缩放因子，通常为 2*r
                target_modules=["query", "value"],  # 针对 ViT 层
                lora_dropout=0.05,  # dropout 防止过拟合
                bias="none",  # 不调整偏置
                task_type=None  # 自定义任务类型
            )

        self.models["depth"] = get_peft_model(model, lora_config)
        # self.low_lr_parameters += list(self.models["depth"].parameters())

        self.models["reflection"] = networks.SpatioTemporalDeformableReflectionModule(
            num_heads=self.opt.heads, 
            embed_dim=self.opt.embed_dim,
            depth=self.opt.str_depth,
            T=len(self.opt.frame_ids),
            img_shape=(self.opt.height, self.opt.width),
            drop_rate=self.opt.drop_rate,
            attn_drop_rate=self.opt.attn_drop_rate,
            drop_path_rate=self.opt.drop_path_rate,
            patch_size=self.opt.patch_size,
            qkv_bias=self.opt.qkv_bias)
        self.normal_lr_parameters += list(self.models["reflection"].parameters())

        # Use the new non-Lambertian decompose decoder
        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        
        self.models["decompose"] = networks.DecomposeDecoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)

        if not self.opt.load_weights_folder:
            model_weights = {
                "decompose": "decompose.pth",
                "decompose_encoder": "decompose_encoder.pth"
            }
            for model_name, weight_name in model_weights.items():
                model_dict = self.models[model_name].state_dict()
                pretrained_dict = torch.load(os.path.join(self.opt.decompose_weights_folder, weight_name), map_location=self.device)
                # 过滤掉不匹配的键，只保留当前模型中存在的参数
                filtered_dict = {k: v for k, v in pretrained_dict.items() \
                                if k in model_dict and v.shape == model_dict[k].shape}
                self.models[model_name].load_state_dict(filtered_dict, strict=False)
                for param in self.models[model_name].parameters():
                    param.requires_grad = False

        # self.low_lr_parameters += list(self.models["decompose_encoder"].parameters())
        # self.low_lr_parameters += list(self.models["decompose"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        # self.normal_lr_parameters += list(self.models["pose_encoder"].parameters())
        self.low_lr_parameters += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.normal_lr_parameters += list(self.models["pose"].parameters())

        for model_name in self.models.keys():
            self.models[model_name].to(self.device)

        self.low_lr_optimizer = optim.Adam(self.low_lr_parameters, self.opt.lora_lr)
        self.model_optimizer = optim.Adam(self.normal_lr_parameters, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, [self.opt.scheduler_step_size], 0.1)
        self.low_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.low_lr_optimizer, [self.opt.scheduler_step_size], 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

    def data_init(self):
        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = utils.readlines(fpath.format("train"))
        val_filenames = utils.readlines(fpath.format("val"))
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=min(self.opt.num_workers, 6),
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        
        self.T = len(self.opt.frame_ids)

    def util_init(self):
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.factor_choicer = utils.FactorChoicer(self.opt.batch_size, self.device)
        self.nabla = Nabla(self.device)

    def set_train(self):
        for model_name in self.models:
            self.models[model_name].train()

    def set_eval(self):
        for model_name in self.models:
            self.models[model_name].eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        if self.opt.load_weights_folder is not None:
            # 从文件夹路径提取epoch数
            try:
                folder_name = os.path.basename(self.opt.load_weights_folder)
                if folder_name.startswith("weights_"):
                    epoch_str = folder_name.replace("weights_", "")
                    self.epoch = int(epoch_str) + 1  # 从下一个epoch开始
                    print("Resuming training from epoch {} (extracted from folder {})".format(self.epoch, folder_name))
                else:
                    try:
                        with open(os.path.join(self.opt.load_weights_folder, "epoch.txt"), "r") as f:
                            self.epoch = int(f.read()) + 1  # 从下一个epoch开始
                        print("Resuming training from epoch {} (from epoch.txt)".format(self.epoch))
                    except FileNotFoundError:
                        print("No epoch file found and folder format not recognized, starting from epoch 0")
                        self.epoch = 0
            except ValueError:
                print("Failed to extract epoch from folder name, starting from epoch 0")
                self.epoch = 0
        
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])   
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # depth, pose, decompose
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            self.low_lr_optimizer.zero_grad()
            losses["loss"].backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.normal_lr_parameters, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.low_lr_parameters, max_norm=1.0)
            
            self.model_optimizer.step()
            self.low_lr_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()
        self.low_lr_scheduler.step()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device, non_blocking=True)  # 异步传输，减少等待时间
        outputs = {}
        outputs[("disp", 0)] = self.models["depth"](inputs["color_aug", 0, 0])
        outputs.update(self.predict_poses(inputs))
        self.decompose(inputs, outputs)
        self.suppress(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
         
        return outputs

    def decompose(self, inputs, outputs):
        """Decompose the input image into albedo, specular, and diffuse components"""
        decompose_features = {}
        for f_i in self.opt.frame_ids:
            decompose_features[f_i] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
            # 随机取增强方式为-1或者1
            factor = self.factor_choicer.get_factor(torch.randint(0, 2, ()).item() * 2 - 1)
            inputs[("color_aug", f_i, 0, "enhanced")] = utils.enhance_brightness_torch(inputs[("color_aug", f_i, 0)], factor)
            decompose_features[(f_i, "enhanced")] = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0, "enhanced")])

        for f_i in self.opt.frame_ids:
            outputs[("decompose_result", f_i)] = self.models["decompose"](decompose_features[f_i], inputs[("color_aug", f_i, 0)])
            outputs[("decompose_result", f_i, "enhanced")] = self.models["decompose"](decompose_features[(f_i, "enhanced")], inputs[("color_aug", f_i, 0, "enhanced")])
            
    def suppress(self, inputs, outputs):
        # 与最初版本相比，先对A和M进行warp，再计算suppress。
        input_A = [outputs[("decompose_result", 0)]["A"]]
        input_M = [outputs[("decompose_result", 0)]["M"]]

        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = utils.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        outputs[("depth", 0)] = depth

        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, f_i)]
            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)

            outputs[("warp", 0, f_i)] = pix_coords

            outputs[("A_warp", f_i)] = F.grid_sample(
                outputs[("decompose_result", f_i)]["A"],
                pix_coords,
                padding_mode="border",
                align_corners=True
            )
            outputs[("M_warp", f_i)] = F.grid_sample(
                outputs[("decompose_result", f_i)]["M"],
                pix_coords,
                padding_mode="border",
                align_corners=True
            )
            # masking zero values
            mask_ones = torch.ones_like(inputs[("color_aug", f_i, 0)])
            mask_warp = F.grid_sample(
                mask_ones,
                outputs[("warp", 0, f_i)],
                padding_mode="zeros", align_corners=True)
            valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
            outputs[("valid_mask", 0, f_i)] = valid_mask
            input_A.append(outputs[("A_warp", f_i)])
            input_M.append(outputs[("M_warp", f_i)])

        # (B, T, C, H, W)
        input_A = torch.stack(input_A, dim=1)
        input_M = torch.stack(input_M, dim=1)
        outputs[("sequenced_M", 0)] = input_M
        outputs[("suppressed_result", "all")] = self.models["reflection"](input_A, input_M)
        for i, f_i in enumerate(self.opt.frame_ids):
            outputs[("suppressed", f_i)] = outputs[("suppressed_result", "all")][:, i, :, :, :]

    def compute_decompose_loss(self, inputs, outputs, losses):
        recons_loss = torch.tensor(0.0, device=self.device)
        # 重构损失
        for f_i in self.opt.frame_ids:
            # 确保当前帧的重构结果
            recons_loss += (
                self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i)]["A"] * outputs[("decompose_result", f_i)]["S"], 
                    inputs[("color_aug", f_i, 0)]) + 
                self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i, "enhanced")]["A"] * (1 - outputs[("decompose_result", f_i, "enhanced")]["S"]), 
                    inputs[("color_aug", f_i, 0, "enhanced")])
            ).mean() / 2
            # 确保当前帧的增强有效
            recons_loss += (
                self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i, "enhanced")]["A"] * outputs[("decompose_result", f_i)]["S"], 
                    inputs[("color_aug", f_i, 0)]) + 
                self.compute_reprojection_loss(
                    outputs[("decompose_result", f_i)]["A"] * (1 - outputs[("decompose_result", f_i, "enhanced")]["S"]), 
                    inputs[("color_aug", f_i, 0, "enhanced")])
            ).mean() / 2
        recons_loss /= self.num_input_frames
        
        # 检查重构损失
        losses["recons_loss"] = recons_loss
        losses["loss"] += recons_loss * self.opt.recons_weight

        # Retinex 损失 
        retinex_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i)]["M"]
            retinex_loss += (self.compute_reprojection_loss(
                self.nabla(outputs[("decompose_result", f_i)]["A"]),
                self.nabla(inputs[("color_aug", f_i, 0)]) * (1 - M)
            )).mean()
        retinex_loss /= self.num_input_frames
        
        # 检查Retinex损失
        losses["retinex_loss"] = retinex_loss
        losses["loss"] += retinex_loss * self.opt.retinex_weight

        # Shading 平滑损失
        smooth_S = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            smooth_S += torch.mean(self.nabla(outputs[("decompose_result", f_i)]["S"]) ** 2)
        smooth_S /= self.num_input_frames
        
        # 检查平滑损失
        losses["smooth_S"] = smooth_S
        losses["loss"] += smooth_S * self.opt.S_smooth_weight

    def sl1(self, inputs, outputs, losses):
        features = self.models["reflection"].features  # List of 4

        # patches: (B*T, N, encoder_embed_dim)
        feat = features[-1]
        B = feat[1].shape[0]
        feat = feat[0]
        BT, N, D = feat.shape
        T = len(self.opt.frame_ids)

        feat = rearrange(feat, '(b t) n d -> b n t d', n=N, t=T)

        # 中间帧
        mid_idx = self.opt.frame_ids.index(0)  # 0 是中间

        # 高光 mask (中间帧)
        M_mid = outputs[("sequenced_M", 0)][:, mid_idx, 0]  # (B, H, W)
        Hp = self.opt.height // self.opt.patch_size
        Wp = self.opt.width // self.opt.patch_size
        M_spec = F.interpolate(M_mid.unsqueeze(1), size=(Hp, Wp), mode='bilinear', align_corners=False).squeeze(1)  # (B, Hp, Wp)
        med = M_spec.mean()
        M_spec = (M_spec > med).float().view(B, -1)  # (B, N)
        feat_mid = feat[:, :, mid_idx, :]  # (B, N, D)

        # 跨帧相似度（前后帧 feat 对比）
        sims = []
        for t in range(T):
            if t == mid_idx:
                continue
            feat_t = feat[:, :, t, :]
            sim = F.cosine_similarity(feat_mid, feat_t, dim=-1)  # (B, N)
            sims.append(sim)
        sim_cross = torch.stack(sims, dim=-1).mean(-1)  # (B, N)

        loss_decouple = (sim_cross * M_spec).mean()  # 高光区相似越高越惩罚
        loss_couple = ((1 - sim_cross) * (1 - M_spec)).mean()  # 非高光区相似越低越惩罚

        loss_feat_temp = loss_decouple + 0.8 * loss_couple
        losses["loss"] += 0.2 * loss_feat_temp

        # 抑制到非高光平均水平
        # 非高光平均特征
        non_spec_mask = (1 - M_spec).unsqueeze(-1)  # (B, N, 1)
        num_non_spec = non_spec_mask.sum(1, keepdim=True)  # (B, 1, 1)
        mean_non_spec = (feat_mid * non_spec_mask).sum(1, keepdim=True) / (num_non_spec + 1e-6)  # (B, 1, D)
        # 高光区域拉近到这个平均
        spec_mask = M_spec.unsqueeze(-1)  # (B, N, 1)
        feat_spec = feat_mid * spec_mask
        loss_avg_suppress = F.l1_loss(feat_spec, mean_non_spec.repeat(1, N, 1) * spec_mask)  # 只在高光区域算
        losses["loss"] += 0.4 * loss_avg_suppress

        # 清理（防止显存）
        del feat, sims, sim_cross

    def sl2(self, inputs, outputs, losses):
        # 2. 重头戏：动态选择性融合损失（真正用上 Temporal Attention）
        # 从你的 sup_head 最后输出的特征（你已经有 intermediate features）
        features = self.models["reflection"].features  # List[4]
        feat = features[-1][0]  # (B*T, N, D)  patches feature
        B, C, H, W = outputs[("suppressed", 0)].shape
        T = len(self.opt.frame_ids)
        BT, N, D = feat.shape
        feat = rearrange(feat, '(b t) n d -> b t n d', b=B, t=T)  # (B, T, N, D)

        mid_idx = self.opt.frame_ids.index(0)
        feat_mid = feat[:, mid_idx]  # (B, N, D)

        # 高光 mask
        # 高光 mask (B, N)
        Hp = self.opt.height // self.opt.patch_size
        Wp = self.opt.width // self.opt.patch_size
        M_mid = outputs[("sequenced_M", 0)][:, mid_idx, 0]  # (B, H, W)
        M_spec = F.interpolate(M_mid.unsqueeze(1), size=(Hp, Wp), mode='bilinear', align_corners=False)
        M_spec = (M_spec.squeeze(1) > 0.7).float().view(B, -1)  # (B, N)

        # 计算中间帧与每一其他帧的相似度
        other_idx = [i for i in range(T) if i != mid_idx]
        sim_to_others = []
        for t in other_idx:
            sim = F.cosine_similarity(feat_mid, feat[:, t], dim=-1)  # (B, N)
            sim_to_others.append(sim)
        sim_to_others = torch.stack(sim_to_others, dim=1)  # (B, T-1, N)

        # 选相似度最高的帧
        best_sim, best_idx = sim_to_others.max(1)  # best_sim/best_idx: (B, N)

        # 损失1：高光区域必须能找到一个相似度高的帧
        loss_find_good = F.relu(0.4 - best_sim) * M_spec
        loss_find_good = loss_find_good.mean()

        # 损失2：选中的帧不能是高光帧（防止选错闪烁帧）
        loss_wrong_choice = 0.0
        feat_other = feat[:, other_idx, :, :]  # (B, T-1, N, D)
        for i, t in enumerate(other_idx):
            chosen = (best_idx == i).float()  # (B, N)
            M_other = outputs[("sequenced_M", 0)][:, t, 0]  # (B, H, W)
            M_other_resized = F.interpolate(M_other.unsqueeze(1), size=(Hp, Wp),
                                            mode='bilinear', align_corners=False).squeeze(1)
            M_other_flat = (M_other_resized > 0.7).float().view(B, -1)  # (B, N)
            loss_wrong_choice += (chosen * M_other_flat * M_spec).mean()

        # 选中的特征拉近当前特征（显式抑制）
        best_idx_exp = best_idx.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        best_idx_exp = best_idx_exp.expand(-1, -1, -1, D)    # (B, 1, N, D)
        chosen_feat = torch.gather(feat_other, 1, best_idx_exp)  # (B, 1, N, D)
        chosen_feat = chosen_feat.squeeze(1)  # (B, N, D)

        loss_suppress = F.l1_loss(
            feat_mid * M_spec.unsqueeze(-1),
            chosen_feat * M_spec.unsqueeze(-1)
        )

        # 加到总损失
        losses["loss"] += 0.8 * loss_find_good
        losses["loss"] += 1.5 * loss_wrong_choice
        losses["loss"] += 1.8 * loss_suppress

    def sl3(self, inputs, outputs, losses):
        features = self.models["reflection"].features[-1][0]  # (B*T, N, D)
        B, C, H, W = outputs[("suppressed", 0)].shape
        T = len(self.opt.frame_ids)
        BT, N, D = features.shape
        features = rearrange(features, '(b t) n d -> b t n d', b=B, t=T)  # (B, T, N, D)

        mid_idx = self.opt.frame_ids.index(0)
        feat_mid = features[:, mid_idx]  # 当前帧特征 (B, N, D)

        # 高光 mask（所有帧）
        M_seq = outputs[("sequenced_M", 0)][:, :, 0]  # (B, T, H, W)
        Hp = H // self.opt.patch_size
        Wp = W // self.opt.patch_size
        M_seq = M_seq.reshape(B*T, 1, H, W)  # (B*T, 1, H, W)
        M_seq_patched = F.interpolate(M_seq, size=(Hp, Wp), mode='bilinear', align_corners=False)  # (B*T, 1, Hp, Wp)
        M_seq_patched = M_seq_patched.view(B, T, Hp, Wp)  # (B, T, Hp, Wp)
        M_seq_patched = M_seq_patched.squeeze(2)  # (B, T, Hp, Wp)
        M_seq_flat = M_seq_patched.view(B, T, -1)  # (B, T, N)
        mid = M_seq_flat.mean()
        M_spec_all = (M_seq_flat > mid).float()  # (B, T, N)

        # 当前帧高光 patch
        M_spec_mid = M_spec_all[:, mid_idx]  # (B, N)

        # 只考虑其他帧
        other_idx = [i for i in range(T) if i != mid_idx]
        feat_other = features[:, other_idx]  # (B, T-1, N, D)
        M_other = M_spec_all[:, other_idx]  # (B, T-1, N)

        # 计算相似度（只在其他帧非高光的位置计算！）
        sims = []
        for i in range(len(other_idx)):
            # 掩掉其他帧的高光区域，只用干净 patch 计算相似度
            clean_mask = (1 - M_other[:, i])  # (B, N)  1=干净
            sim = F.cosine_similarity(feat_mid, feat_other[:, i], dim=-1)  # (B, N)
            sim = sim * clean_mask + (-1.0) * M_other[:, i]  # 高光区域强制低相似度
            sims.append(sim)
        sims = torch.stack(sims, dim=1)  # (B, T-1, N)

        # 选相似度最高的干净帧
        best_sim, best_idx = sims.max(1)  # (B, N)

        # 损失1：高光区域必须找到一个高相似度的干净帧
        loss_must_find = F.relu(0.5 - best_sim) * M_spec_mid
        loss_must_find = loss_must_find.mean()

        # 损失2：选中的帧必须真的是干净的（双保险）
        chosen_clean = torch.gather(1 - M_other, 1, best_idx.unsqueeze(1)).squeeze(1)  # (B, N)
        loss_selected_clean = (1 - chosen_clean) * M_spec_mid
        loss_selected_clean = loss_selected_clean.mean()

        # 损失3：把当前高光 patch 的特征拉向选中的干净特征
        best_idx_exp = best_idx.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, D)  # (B, 1, N, D)
        chosen_feat = torch.gather(feat_other, 1, best_idx_exp).squeeze(1)  # (B, N, D)

        loss_fill_clean = F.l1_loss(
            feat_mid * M_spec_mid.unsqueeze(-1),
            chosen_feat * M_spec_mid.unsqueeze(-1)
        )

        # 加权求和（重点权重在 fill_clean）
        losses["loss"] += 0.8 * loss_must_find
        losses["loss"] += 1.5 * loss_selected_clean
        losses["loss"] += 2.0 * loss_fill_clean  # 最重要！强制填充干净特征

        # losses["must_find"] = loss_must_find.item()
        # losses["selected_clean"] = loss_selected_clean.item()
        # losses["fill_clean"] = loss_fill_clean.item()

    def sl4(self, inputs, outputs, losses):
        features = self.models["reflection"].features[-1][0]  # (B*T, N, D)
        B, C, H, W = outputs[("suppressed", 0)].shape
        T = len(self.opt.frame_ids)
        BT, N, D = features.shape
        features = rearrange(features, '(b t) n d -> b t n d', b=B, t=T)  # (B, T, N, D)

        mid_idx = self.opt.frame_ids.index(0)
        feat_mid = features[:, mid_idx]  # (B, N, D)
        M_seq = outputs[("sequenced_M", 0)][:, :, 0]  # (B, T, H, W)
        Hp = H // self.opt.patch_size
        Wp = W // self.opt.patch_size
        M_seq = M_seq.reshape(B * T, 1, H, W)
        M_seq_patched = F.interpolate(M_seq, size=(Hp, Wp), mode='bilinear', align_corners=False)  # (B*T, 1, Hp, Wp)
        M_seq_patched = M_seq_patched.view(B, T, Hp, Wp)
        M_seq_flat = M_seq_patched.view(B, T, -1)  # (B, T, N)
        threshold = M_seq_flat.mean()
        M_spec_all = (M_seq_flat > threshold).float()  # (B, T, N)

        M_spec_mid = M_spec_all[:, mid_idx]  # (B, N)

        # 其他帧
        other_idx = [i for i in range(T) if i != mid_idx]
        K = len(other_idx)
        feat_other = features[:, other_idx]  # (B, K, N, D)
        M_other = M_spec_all[:, other_idx]  # (B, K, N)

        # 向量化 3x3 邻域展开
        feat_other_grid = feat_other.view(B * K, Hp, Wp, D).permute(0, 3, 1, 2)  # (B*K, D, Hp, Wp)
        unfold_feat = F.unfold(feat_other_grid, kernel_size=3, padding=1)  # (B*K, D*9, N)
        unfold_feat = unfold_feat.view(B, K, D, 9, N).permute(0, 1, 4, 3, 2)  # (B, K, N, 9, D)

        M_other_grid = M_other.view(B * K, Hp, Wp).unsqueeze(1)  # (B*K, 1, Hp, Wp)
        unfold_M = F.unfold(M_other_grid, kernel_size=3, padding=1)  # (B*K, 9, N)
        unfold_M = unfold_M.view(B, K, 9, N).permute(0, 1, 3, 2)  # (B, K, N, 9)

        # 当前 patch 特征重复
        cur_feat = feat_mid.unsqueeze(1).unsqueeze(3).repeat(1, K, 1, 9, 1)  # (B, K, N, 9, D)

        # 相似度
        sims = F.cosine_similarity(cur_feat, unfold_feat, dim=-1)  # (B, K, N, 9)

        # 只用干净邻域
        clean_mask = (1 - unfold_M)  # (B, K, N, 9)
        sims = sims * clean_mask - unfold_M  # 高光区域给 -1

        # 选最佳邻域
        best_sim_per_k, _ = sims.max(-1)  # (B, K, N)
        best_sim, best_k = best_sim_per_k.max(1)  # (B, N)

        # fallback_mask：如果所有邻域都不干净或相似度太低
        fallback_mask = (best_sim < 0.3).float()  # (B, N)   ← 修复了你报错的这行

        # 选中的特征
        best_k_exp = best_k.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 9, D)  # (B, 1, N, 9, D)
        best_neigh_feat = torch.gather(unfold_feat, 1, best_k_exp).squeeze(1)  # (B, N, 9, D)
        best_neigh_sim, best_neigh_idx = sims.gather(1, best_k.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 9)).squeeze(1).max(-1)  # (B, N)
        best_neigh_idx = best_neigh_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, D)
        chosen_feat = torch.gather(best_neigh_feat, 2, best_neigh_idx).squeeze(2)  # (B, N, D)

        # fallback：全局非高光平均
        non_spec = (1 - M_spec_mid).unsqueeze(-1)  # (B, N, 1)
        num_non = non_spec.sum(1, keepdim=True).clamp(min=1)
        mean_non = (feat_mid * non_spec).sum(1, keepdim=True) / num_non  # (B, 1, D)
        chosen_feat = chosen_feat * (1 - fallback_mask.unsqueeze(-1)) + mean_non.repeat(1, N, 1) * fallback_mask.unsqueeze(-1)

        # 抑制损失
        loss_fill = F.mse_loss(feat_mid * M_spec_mid.unsqueeze(-1), chosen_feat * M_spec_mid.unsqueeze(-1))
        loss_find = F.relu(0.4 - best_sim) * M_spec_mid
        loss_find = loss_find.mean()

        losses["loss"] += 1.5 * loss_fill + 0.8 * loss_find

    def compute_supressed_loss(self, inputs, outputs, losses):
        self.sl2(inputs, outputs, losses)

        sim_loss = torch.tensor(0.0, device=self.device)
        if self.epoch < self.opt.str_sim_epoch:
            for f_i in self.opt.frame_ids:
                sim_loss += self.compute_reprojection_loss(
                    outputs[("suppressed", f_i)],
                    outputs[("decompose_result", f_i)]["A"]
                ).mean()
            sim_loss /= self.num_input_frames
            losses["sim_loss"] = sim_loss
            losses["loss"] += sim_loss
        else:
            for f_i in self.opt.frame_ids[1:]:
                M = outputs[("M_warp", f_i)]
                med = M.mean()
                B = self.opt.batch_size
                M_spec = (M > med).float().view(B, 1, self.opt.height, self.opt.width)
                sim_loss += self.compute_reprojection_loss(
                    outputs[("suppressed", f_i)] * (1 - M_spec),
                    outputs[("A_warp", f_i)] * (1 - M_spec)
                ).mean()
            sim_loss /= self.num_input_frames
            losses["sim_loss"] = sim_loss
            losses["loss"] += sim_loss * 1.5

        if self.epoch >= self.opt.str_sim_epoch:
            reprojection_loss = torch.tensor(0.0, device=self.device)
            for f_i in self.opt.frame_ids[1:]:
                mask = outputs[("valid_mask", 0, f_i)]  
                mask_sum = mask.sum()
                if mask_sum > 0:
                    reprojection_loss += self.compute_reprojection_loss(
                        outputs[("suppressed", f_i)],
                        outputs[("suppressed", 0)]
                    ).sum() / mask_sum
            reprojection_loss /= (self.num_input_frames - 1)
            losses["reprojection_loss"] = reprojection_loss
            losses["loss"] += reprojection_loss * self.opt.reprojection_weight

        if self.epoch == 0 or ((self.opt.load_weights_folder is not None) and (self.epoch < self.opt.str_sim_epoch)):
            for param in self.models["depth"].parameters():
                param.requires_grad = False
        if self.epoch == self.opt.str_sim_epoch:
            for param in self.models["depth"].parameters():
                param.requires_grad = True

        if self.epoch >= self.opt.str_sim_epoch:
            disp = outputs[("disp", 0)]  
            color = inputs[("color_aug", 0, 0)]  
            mean_disp = disp.mean(2, True).mean(3, True)  
            norm_disp = disp / (mean_disp + 1e-4)  
            norm_disp = torch.clamp(norm_disp, min=0.1, max=10.0)
            loss_disp_smooth = get_smooth_loss(norm_disp, color)  
            losses["disp_smooth_loss"] = loss_disp_smooth
            losses["loss"] += loss_disp_smooth * self.opt.disp_smooth_weight
        
        # 使用梯度，在非高光区域上抑制伪影
        for f_i in self.opt.frame_ids:
            M = outputs[("decompose_result", f_i)]["M"]
            med = M.mean()
            B = self.opt.batch_size
            M_spec = (M > med).float().view(B, 1, self.opt.height, self.opt.width)
            sim_loss += self.compute_reprojection_loss(
                self.nabla(outputs[("suppressed", f_i)]) * (1 - M_spec),
                self.nabla(outputs[("decompose_result", f_i)]["A"]) * (1 - M_spec)
            ).mean()
        sim_loss /= self.num_input_frames
        losses["sim_loss"] = sim_loss
        losses["loss"] += sim_loss * 1.5

    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = torch.tensor(0.0, device=self.device)

        # 计算分解的损失
        # self.compute_decompose_loss(inputs, outputs, losses)
        
        # 计算抑制后的损失
        self.compute_supressed_loss(inputs, outputs, losses)
        
        return losses

    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（重建损失）
        
        使用SSIM+L1混合损失，平衡结构相似性和像素级准确性
        
        Args:
            pred: 预测图像（重建图像）
            target: 目标图像（真实图像）
            
        Returns:
            reprojection_loss: 逐像素的重投影损失
        """
        # 添加数值稳定性保护
        pred = torch.clamp(pred, min=1e-6, max=1.0-1e-6)
        target = torch.clamp(target, min=1e-6, max=1.0-1e-6)
        
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        
        # 确保SSIM损失在有效范围内
        ssim_loss = torch.clamp(ssim_loss, min=0.0, max=1.0)
        
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
            # 清理GPU缓存，避免内存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 关键修复：清理reflection模块的attention maps
            if hasattr(self.models["reflection"], "clear_temporal_attention_maps"):
                self.models["reflection"].clear_temporal_attention_maps()
        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  utils.sec_to_hm_str(time_sofar), utils.sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for l, v in losses.items():
            # 在记录损失之前检查NaN/Inf，如果检测到则直接退出程序
            if torch.isinf(v).any():
                print(f"\n[ERROR] TensorBoard日志记录时检测到inf值，程序即将退出...")
                print(f"具体损失函数: {l}")
                print(f"损失值: {v}")
                print(f"问题位置: log() 方法 - 模式: {mode}")
                print(f"\n[EXIT] 由于检测到inf值，程序异常退出")
                import sys
                sys.exit(1)
            elif torch.isnan(v).any():
                print(f"[WARNING] TensorBoard日志记录: 损失 {l} 包含NaN值: {v}")
                # 使用安全的数值替代
                v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            # 在记录图像之前检查NaN/Inf
            try:
                disp_data = outputs[("disp", 0)][j]
                if torch.isnan(disp_data).any() or torch.isinf(disp_data).any():
                    print(f"[WARNING] TensorBoard日志记录: disp图像包含NaN或Inf值，使用第{j}个样本")
                    disp_data = torch.nan_to_num(disp_data, nan=0.0, posinf=1.0, neginf=0.0)
                writer.add_image("disp/{}".format(j), utils.visualize_depth(disp_data), self.step)
                
                input_data = inputs[("color_aug", 0, 0)][j].data
                if torch.isnan(input_data).any() or torch.isinf(input_data).any():
                    print(f"[WARNING] TensorBoard日志记录: input图像包含NaN或Inf值，使用第{j}个样本")
                    input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=0.0)
                writer.add_image("input/{}".format(j), input_data, self.step)
                
                for frame_id in self.opt.frame_ids:
                    writer.add_image("suppressed {}/{}".format(frame_id, j), outputs[("suppressed", frame_id)][j].data, self.step)
                
                # A_data = outputs[("decompose_result", 0)]["A"][j].data
                # if torch.isnan(A_data).any() or torch.isinf(A_data).any():
                #     print(f"[WARNING] TensorBoard日志记录: A图像包含NaN或Inf值，使用第{j}个样本")
                #     A_data = torch.nan_to_num(A_data, nan=0.0, posinf=1.0, neginf=0.0)
                # writer.add_image("A/{}".format(j), A_data, self.step)
                for frame_id in self.opt.frame_ids:
                    writer.add_image("A {}/{}".format(frame_id, j), outputs[("decompose_result", frame_id)]["A"][j].data, self.step)
                for frame_id in self.opt.frame_ids[1:]:
                    writer.add_image("A_warp {}/{}".format(frame_id, j), outputs[("A_warp", frame_id)][j].data, self.step)

                for frame_id in self.opt.frame_ids[1:]:
                    writer.add_image("valid mask {}/{}".format(frame_id, j), outputs[("valid_mask", 0, frame_id)][j].data, self.step)
                
                S_data = outputs[("decompose_result", 0)]["S"][j].data
                if torch.isnan(S_data).any() or torch.isinf(S_data).any():
                    print(f"[WARNING] TensorBoard日志记录: S图像包含NaN或Inf值，使用第{j}个样本")
                    S_data = torch.nan_to_num(S_data, nan=0.0, posinf=1.0, neginf=0.0)
                writer.add_image("S/{}".format(j), S_data, self.step)
                
                M_data = outputs[("decompose_result", 0)]["M"][j].data
                if torch.isnan(M_data).any() or torch.isinf(M_data).any():
                    print(f"[WARNING] TensorBoard日志记录: M图像包含NaN或Inf值，使用第{j}个样本")
                    M_data = torch.nan_to_num(M_data, nan=0.0, posinf=1.0, neginf=0.0)
                writer.add_image("M/{}".format(j), utils.visualize_depth(M_data, cmap='plasma'), self.step)


            except Exception as e:
                print(f"[ERROR] TensorBoard日志记录失败: {e}")
                continue
            
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
        # Save the current epoch number
        epoch_path = os.path.join(save_folder, "epoch.txt")
        with open(epoch_path, "w") as f:
            f.write(str(self.epoch))

        if hasattr(self.models["reflection"], "remove_temporal_hooks"):
                self.models["reflection"].remove_temporal_hooks()

    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        if self.opt.models_to_load is not None:
            for n in self.opt.models_to_load:
                print("Loading {} weights...".format(n))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
        
        # Load optimizer state if it exists
        optimizer_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.exists(optimizer_path):
            try:
                optimizer_dict = torch.load(optimizer_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
                print("Loaded optimizer state")
            except Exception as e:
                print("Failed to load optimizer state: {}".format(e))
        else:
            print("No optimizer state found")