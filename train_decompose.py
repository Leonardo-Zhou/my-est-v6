from __future__ import absolute_import, division, print_function

import sys
import os
from trainer_decompose import Trainer
from options import Options

# 调试模式开关
DEBUG_MODE = True  # 设置为True启用调试模式

def setup_debug_args():
    """设置调试用的参数"""
    options = Options()
    opts = options.parse()
    
    if DEBUG_MODE:
        print("🐛 调试模式已启用，使用预设参数...")
        # opts.load_weights_folder = "./decompose_ckpt/decompose/models/weights_5"
        opts.models_to_load = ["decompose_encoder", "decompose"]
        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        opts.model_name = f'decompose'
        opts.log_dir = "./decompose_ckpt"
        opts.num_epochs = 20
        opts.batch_size = 6
        opts.scheduler_step_size = 3
    return opts

if __name__ == "__main__":
    # 获取调试参数
    opts = setup_debug_args()
    # 创建训练器并开始训练
    trainer = Trainer(opts)
    trainer.train()