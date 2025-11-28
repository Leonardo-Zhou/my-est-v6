from __future__ import absolute_import, division, print_function

import sys
import os
from trainer_masked import Trainer
from options import Options

# 调试模式开关
DEBUG_MODE = True  # 设置为True启用调试模式

def setup_debug_args():
    """设置调试用的参数"""
    options = Options()
    opts = options.parse()
    
    if DEBUG_MODE:
        print("🐛 调试模式已启用，使用预设参数...")
        
        # 直接设置参数值
        opts.load_weights_folder = "./logs_masked/str_sim/models/weights_9"
        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        opts.decompose_weights_folder = "./decompose_ckpt/decompose_new1/models/weights_14"
        opts.models_to_load = ["pose_encoder", "pose", "decompose_encoder", "decompose", 'reflection']
        opts.log_dir = "./logs_masked"
        opts.model_name = "suppress_A_reprojection"
        opts.description = "使用MaskedSpatioTemporalReflectionModule，进行高光抑制。认为，当前帧的高光区域在前后帧，甚至是前后数十帧上可能依旧是高光区域，无法从其中提取特征用于重建。因此考虑抑制，而非特种重新的补全。"
        opts.num_epochs = 20
        opts.batch_size = 4
        opts.patch_size = 16
        opts.str_depth = 12

        # 可以添加更多调试友好的参数
        # opts.num_workers = 1  # 单线程，便于调试
        # opts.log_frequency = 1  # 更频繁的日志输出

    return opts
if __name__ == "__main__":
    # 获取调试参数
    opts = setup_debug_args()
    
    # 创建训练器并开始训练
    trainer = Trainer(opts)
    trainer.train() 