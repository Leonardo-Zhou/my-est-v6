from __future__ import absolute_import, division, print_function

import sys
import os
from trainer import Trainer
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
        # opts.load_weights_folder = "./logs_v2/Change2+4/models/weights_21"
        opts.data_path = "/data2/publicData/MICCAI19_SCARED/train"
        opts.decompose_weights_folder = "./decompose_ckpt/decompose/models/weights_5"
        opts.log_dir = "./logs"
        opts.model_name = "vits"
        opts.num_epochs = 30
        opts.batch_size = 8
        
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