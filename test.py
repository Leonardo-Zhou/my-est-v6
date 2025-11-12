import torch
import matplotlib.pyplot as plt
import networks
from visualize_utils import *
import torch.nn.functional as F
import gc
# 显存监控函数
def get_memory_info():
    """获取当前显存使用情况"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,     # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

def print_memory_info(prefix=""):
    """打印显存信息"""
    if torch.cuda.is_available():
        memory_info = get_memory_info()
        print(f"{prefix}显存: 已分配={memory_info['allocated']:.3f}GB, "
              f"缓存={memory_info['cached']:.3f}GB, 峰值={memory_info['max_allocated']:.3f}GB")

def clear_memory():
    """清理显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def analyze_model_memory(model, model_name="Model"):
    """分析模型的参数显存占用"""
    if not torch.cuda.is_available():
        return 0
    
    param_size = 0
    buffer_size = 0
    
    # 计算参数大小
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # 计算缓冲区大小
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    total_size_gb = total_size / 1024**3
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{model_name} 参数分析:")
    print(f"  总参数量: {param_count:,}")
    print(f"  可训练参数量: {trainable_param_count:,}")
    print(f"  参数显存占用: {total_size_gb:.3f}GB")
    print(f"  平均参数大小: {(total_size / param_count) if param_count > 0 else 0:.1f} bytes")
    
    return total_size_gb

def extract_samples(suppressed_seq, M_seq, num_pos=128, num_neg=128):
    B, T, C, H, W = suppressed_seq.shape
    flat_feat = suppressed_seq.view(B, T, C, -1)  # [B, T, C, HW]
    flat_M = M_seq.view(B, T, 1, -1)
    mid = T // 2  # 中间帧索引

    # 正样本：相邻帧 (t-1, t+1) 非反射平均 [B, C]
    non_reflect_mask = (flat_M[:, mid-1:mid+2] < 0.5).squeeze(2).float()  # [B, 3, HW]

    # 获取相邻帧的特征 [B, 2, C, HW]
    pos_feat = flat_feat[:, mid-1:mid+2]  # [B, 2, C, HW]
    pos_feat_flat = pos_feat.permute(0, 2, 1, 3).reshape(B, C, -1)  # [B, C, 3*HW]
    
    # 从相邻帧的非反射区域采样
    non_reflect_mask_adj = non_reflect_mask.reshape(B, -1)  # [B, 3*HW]
    pos_indices = torch.multinomial(non_reflect_mask_adj, num_pos, replacement=True)
    
    # 采样并平均
    pos_samples = torch.gather(pos_feat_flat, 2, pos_indices.unsqueeze(1).expand(-1, C, -1))  # [B, C, num_pos]
    pos = pos_samples.mean(dim=2)  # [B, C]

    # 负样本：当前序列反射平均 [B, C]
    reflect_mask = (flat_M > 0.5).squeeze(2).float()  # [B, T, HW]
    reflect_mask_flat = reflect_mask.reshape(B, -1)  # [B, T*HW]
    
    neg_indices = torch.multinomial(reflect_mask_flat, num_neg, replacement=True)
    
    # 获取所有时间步的特征 [B, T*C, HW]
    neg_feat_flat = flat_feat.permute(0, 2, 1, 3).reshape(B, C, -1)  # [B, C, T*HW]
    neg_samples = torch.gather(neg_feat_flat, 2, neg_indices.unsqueeze(1).expand(-1, C, -1))  # [B, C, num_neg]
    neg = neg_samples.mean(dim=2)  # [B, C]
    
    return pos, neg

# 时空对比损失（同前，但时空anchor）
def contrastive_loss(anchor, pos, neg, tau=0.07):
    anchor, pos, neg = map(F.normalize, [anchor, pos, neg])
    sim_pos = F.cosine_similarity(anchor, pos, dim=-1)
    sim_neg = F.cosine_similarity(anchor, neg, dim=-1)
    logits = torch.stack([sim_pos, sim_neg], dim=1) / tau
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)



def test_memory_usage():
    """测试实际网络的显存使用情况"""
    print("=== 实际网络显存分析测试 ===")
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，无法进行显存分析")
        return
    
    # 清理显存
    clear_memory()
    torch.cuda.reset_peak_memory_stats()
    print_memory_info("初始状态: ")
    
    print("\n1. 创建网络结构...")
    
    # 创建编码器
    encoder = networks.ResnetEncoder(18, False).to(torch.device("cuda"))
    encoder_memory = analyze_model_memory(encoder, "ResNet编码器")
    print_memory_info("创建编码器后: ")
    
    # 创建解码器
    decoder = networks.DecomposeDecoder(encoder.num_ch_enc).to(torch.device("cuda"))
    decoder_memory = analyze_model_memory(decoder, "Decompose解码器")
    print_memory_info("创建解码器后: ")

    test_configs = [
        # {'drm': networks.DynamicReflectionModule, 'name': 'DynamicReflectionModule'},
        # {'drm': networks.ViTReflectionModule, 'name': 'VitReflectionModule'},
        {'drm': networks.SpatioTemporalReflectionModule, 'name': 'SpatioTemporalReflectionModule'},
    ]
    for config in test_configs:
        print(f"\n--- 测试网络 {config['name']} ---")

        # 创建DynamicReflectionModule
        drm = config['drm'](patch_size=16, embed_dim=128).to(torch.device("cuda"))
        drm_memory = analyze_model_memory(drm, config['name'])
        print_memory_info("创建DRM后: ")
        
        total_model_memory = encoder_memory + decoder_memory + drm_memory
        print(f"\n总模型参数显存: {total_model_memory:.3f}GB")
        
        # 测试不同配置的显存需求
        print("\n2. 测试数据显存需求...")

        try:
            # 重置峰值显存记录
            torch.cuda.reset_peak_memory_stats()
            
            B, T, H, W = 8, 5, 256, 320
            
            # 创建测试数据
            images = []
            for t in range(T):
                images.append(torch.randn(B, 3, H, W).cuda())
            print_memory_info("创建输入图像后: ")
            
            # 前向传播 - 编码器
            print("\n3. 执行前向传播...")
            feats = []
            for t in range(T):
                feats.append(encoder(images[t]))
            print_memory_info("编码器前向传播后: ")
            
            # 前向传播 - 解码器
            outputs = []
            for t in range(T):
                outputs.append(decoder(feats[t], images[t]))
            print_memory_info("解码器前向传播后: ")
            
            # 准备DRM输入

            drm_input_A = torch.cat([output["A"].unsqueeze(1) for output in outputs], dim=1)
            drm_input_M = torch.cat([output["M"].unsqueeze(1) for output in outputs], dim=1)
            
            # 前向传播 - DRM
            suppressed_seq = drm(drm_input_A, drm_input_M)
            print_memory_info("DRM前向传播后: ")
            
            # 提取正负样本
            pos, neg = extract_samples(suppressed_seq, drm_input_M)
            print_memory_info("提取样本后: ")
            
            # 计算anchor
            anchor = suppressed_seq.mean([3, 4]).mean(1)
            print_memory_info("计算anchor后: ")
            
            # 计算损失
            loss = contrastive_loss(anchor, pos, neg)
            print_memory_info("计算损失后: ")
            
            # 反向传播
            print("\n4. 执行反向传播...")
            loss.backward()
            print_memory_info("反向传播后: ")
            
            # 获取峰值显存
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n峰值显存使用: {peak_memory:.3f}GB")
            
            # 清理梯度
            encoder.zero_grad()
            decoder.zero_grad()
            drm.zero_grad()
            
            print(f"\n--- 配置 {config['name']} 测试完成 ---")
            clear_memory()
            
        except Exception as e:
            print(f"配置 {config['name']} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 最终清理
    clear_memory()
    print_memory_info("最终清理后: ")

# 运行测试
if __name__ == "__main__":
    test_memory_usage()