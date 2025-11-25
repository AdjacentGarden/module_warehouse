import torch
import torch.nn as nn

class ConditionalPositionalEncoding(nn.Module):
    """
    Conditional Positional Encoding (CPE) implementation for 1D Temporal Data.
    Paper: https://arxiv.org/abs/2102.10882
    
    Why this is SOTA for your task:
    1. Dynamic: Handles any video length (perfect for online streaming).
    2. Plug-and-Play: Fits strictly into forward(self, x).
    3. Inductive Bias: Uses Conv1d to inject local temporal awareness.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # Depthwise Convolution: 分组数等于通道数，极大的节省参数，只关注时序邻域
        # padding=kernel_size//2 保证输出长度和输入一致
        self.proj = nn.Conv1d(
            in_channels=dim, 
            out_channels=dim, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2, 
            groups=dim  # key part
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor. 
               Supports both (Batch, Dim, Time) OR (Batch, Time, Dim)
               The code automatically detects and adapts.
        Returns:
            x: Tensor with positional information injected (x + pe).
        """
        B, N, C = x.shape
        
        feat = x.permute(0, 2, 1) # (B, T, D) -> (B, D, T)

        pos_embed = self.proj(feat)

        pos_embed = pos_embed.permute(0, 2, 1) # (B, D, T) -> (B, T, D)
        
        return x + pos_embed
# ! LN Substitute
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    目前 LLaMA, T5, Gopher 等 SOTA 大模型标配的归一化层。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 相比标准 LN，RMSNorm 通常不需要 bias，只有 weight (scale)
        # 这减少了参数量，且实验证明 bias 对性能提升微乎其微甚至有副作用
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 核心公式: x / RMS(x)
        # rsqrt 是 1/sqrt 的快速计算指令
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # x shape: (Batch, Length, Dim)
        
        # 1. 混合精度训练的关键 Trick：
        # 无论输入 x 是 FP16 还是 BF16，计算统计量（RMS）时必须强制转为 FP32 (float)
        # 否则在计算 x^2 时极易发生数值溢出或下溢
        output = self._norm(x.float()).type_as(x)
        
        # 2. 缩放
        return output * self.weight
    
# ! Scale
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 自动适配维度：只在 Batch 维度生成 mask (B, 1, 1...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class LayerScale(nn.Module):
    """
    SOTA LayerScale implementation.
    特点:
    1. 支持 FP32 混合精度保护
    2. 支持 (B, C, L) 和 (B, L, C) 两种格式自动适配
    3. 集成 DropPath
    """
    def __init__(self, dim, init_values=1e-5, inplace=False, data_format="channels_last"):
        super().__init__()
        self.inplace = inplace
        self.data_format = data_format
        
        # 初始化为极小值 (epsilon)，这是让深层网络收敛的关键
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, drop_prob=0.0):
        # x: Input tensor
        # drop_prob: 当前层的 drop path 概率 (通常由外部 schedule 传入)
        
        # 1. 维度适配 (Handling Data Format)
        if self.data_format == "channels_first": 
            # 输入是 (B, C, T) 或 (B, C, H, W)
            # gamma 需要变成 (C, 1) 或 (C, 1, 1) 以便广播
            gamma = self.gamma.view(1, -1, *([1] * (x.ndim - 2)))
        else:
            # 输入是 (B, T, C) - Transformer 标准格式
            gamma = self.gamma
            
        # 2. 混合精度安全计算 (FP32 Safety Cast)
        # 在执行乘法前，确保使用 fp32 进行缩放，这在 deepspeed/amp 训练中非常重要
        # 否则 1e-5 在 fp16 下可能精度丢失
        gamma = gamma.to(dtype=torch.float32)
        x_dtype = x.dtype
        
        # 3. 核心计算: x * scale
        # 先转 FP32 乘，再转回原精度
        if self.inplace:
            # 节省显存
            x = x.mul_(gamma).to(dtype=x_dtype)
        else:
            x = x.mul(gamma).to(dtype=x_dtype)
            
        # 4. Stochastic Depth (DropPath)
        # Scale 层通常是残差分支的最后一站，所以在这里做 DropPath 最合适
        if drop_prob > 0.:
            x = drop_path(x, drop_prob, self.training)
            
        return x