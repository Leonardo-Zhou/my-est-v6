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

        # 初始化LoRA微调参数
        lora_config = LoraConfig(
            r=10,  # 低秩维度，平衡效率和性能
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
        if not self.opt.load_weights_folder:
            model.load_state_dict(torch.load(os.path.join(self.opt.vit_folder, f'depth_anything_v2_{encoder}.pth'), map_location='cuda'))
        self.models["depth"] = get_peft_model(model, lora_config)
        self.low_lr_parameters += list(self.models["depth"].parameters())

        self.models["reflection"] = networks.SpatioTemporalReflectionModule(
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

        self.low_lr_parameters += list(self.models["decompose_encoder"].parameters())
        self.low_lr_parameters += list(self.models["decompose"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.normal_lr_parameters += list(self.models["pose_encoder"].parameters())

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

    def util_init(self):
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.batch_norm = utils.NormalizeImageBatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=self.device)

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
        input_A = [outputs[("decompose_result", f_i)]["A"] for f_i in self.frames]
        input_M = [outputs[("decompose_result", f_i)]["M"] for f_i in self.frames]
        # (B, T, C, H, W)
        input_A = torch.stack(input_A, dim=1)
        input_M = torch.stack(input_M, dim=1)
        outputs[("sequenced_M", 0)] = input_M
        outputs[("suppressed_result", "all")] = self.models["reflection"](input_A, input_M)
        for f_i in self.frames:
            outputs[("suppressed", f_i)] = outputs[("suppressed_result", "all")][:, f_i, :, :, :]

        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = utils.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        outputs[("depth", 0)] = depth

        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, f_i)]
            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)

            outputs[("warp", 0, f_i)] = pix_coords

            outputs[("suppressed_warp", f_i)] = F.grid_sample(
                outputs[("suppressed", f_i)],
                pix_coords,
                mode="bilinear",
                align_corners=False
            )
            # masking zero values
            mask_ones = torch.ones_like(inputs[("color_aug", f_i, 0)])
            mask_warp = F.grid_sample(
                mask_ones,
                outputs[("warp", 0, f_i)],
                padding_mode="zeros", align_corners=True)
            valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
            outputs[("valid_mask", 0, f_i)] = valid_mask

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
        losses["retinex_loss"] = retinex_loss
        losses["loss"] += retinex_loss * self.opt.retinex_weight

        # Shading 平滑损失
        smooth_S = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids:
            smooth_S += torch.mean(self.nabla(outputs[("decompose_result", f_i)]["S"]) ** 2)
        smooth_S /= self.num_input_frames
        losses["smooth_S"] = smooth_S
        losses["loss"] += smooth_S * self.opt.S_smooth_weight

    def compute_supressed_loss(self, inputs, outputs, losses):
        nt_xent_loss = torch.tensor(0.0, device=self.device)
        # NT-Xent 损失
        pos, neg = utils.extract_samples(
            outputs[("suppressed_result", "all")],
            outputs[("sequenced_M", 0)]
        )
        anchor = outputs[("suppressed_result", "all")].mean([3, 4]).mean(1)
        nt_xent_loss += utils.contrastive_loss(anchor, pos, neg)
        losses["nt_xent_loss"] = nt_xent_loss
        losses["loss"] += nt_xent_loss * self.opt.nt_xent_weight

        if self.epoch < 2:
            sim_loss = torch.tensor(0.0, device=self.device)
            for f_i in self.opt.frame_ids:
                sim_loss += self.compute_reprojection_loss(
                    outputs[("suppressed", f_i)],
                    outputs[("decompose_result", f_i)]["A"]
                ).mean()
            sim_loss /= self.num_input_frames
            losses["sim_loss"] = sim_loss
            losses["loss"] += sim_loss * 0.5


        # 计算重投影损失
        reprojection_loss = torch.tensor(0.0, device=self.device)
        for f_i in self.opt.frame_ids[1:]:
            mask = outputs[("valid_mask", 0, f_i)]  # 有效像素掩码（排除遮挡区域）
            mask_sum = mask.sum()
            reprojection_loss += self.compute_reprojection_loss(
                outputs[("suppressed_warp", f_i)],
                outputs[("suppressed", 0)]
            ).sum() / mask_sum
        reprojection_loss /= (self.num_input_frames - 1)
        losses["reprojection_loss"] = reprojection_loss
        losses["loss"] += reprojection_loss * self.opt.reprojection_weight

        # 计算视差平滑性损失
        disp = outputs[("disp", 0)]  # 视差图
        color = inputs[("color_aug", 0, 0)]  # 对应的彩色图像
        mean_disp = disp.mean(2, True).mean(3, True)  # 计算平均视差（归一化用）
        norm_disp = disp / (mean_disp + 1e-7)  # 归一化视差图
        loss_disp_smooth = get_smooth_loss(norm_disp, color)  # 边缘感知的平滑损失
        losses["disp_smooth_loss"] = loss_disp_smooth
        losses["loss"] += loss_disp_smooth * self.opt.disp_smooth_weight

    def compute_losses(self, inputs, outputs):
        losses = {}
        losses["loss"] = torch.tensor(0.0, device=self.device)

        # 计算分解的损失
        self.compute_decompose_loss(inputs, outputs, losses)
        
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
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
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
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            writer.add_image("disp/{}".format(j), utils.visualize_depth(outputs[("disp", 0)][j]), self.step)
            writer.add_image("input/{}".format(j), inputs[("color_aug", 0, 0)][j].data, self.step)
            writer.add_image("suppressed/{}".format(j), outputs[("suppressed", 0)][j].data, self.step)
            writer.add_image("A/{}".format(j), outputs[("decompose_result", 0)]["A"][j].data, self.step)
            writer.add_image("S/{}".format(j), outputs[("decompose_result", 0)]["S"][j].data, self.step)
            writer.add_image("M/{}".format(j), outputs[("decompose_result", 0)]["M"][j].data, self.step)
            

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