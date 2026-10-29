import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ 标准 3x3 Conv + BN + ReLU 块 """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ChannelSqueezeBlock(nn.Module):
    """ Squeeze-and-Excitation 通道注意力模块 """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # 输出 0 到 1 的通道权重
        )

    def forward(self, x):
        """ x: (B, C, H, W) """
        B, C, _, _ = x.shape
        # Squeeze: (B, C, H, W) -> (B, C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B, C) -> (B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # Scale: 通道加权
        return x * y.expand_as(x)


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 3D 卷积：捕获局部空间-光谱相关性 (kernel_size=(3, 1, 1))
        # 注意：这里的输入维度是 (B, P, H, W)，所以 P 是通道数。
        # 但在 U-Net 结构中，通常将 P 视作通道 C，这里我们调整为 3D-Conv 处理 (B, C, H, W)
        # 假设 U-Net 内部的特征图形状为 (B, C, H, W)

        # 简化：使用 2D 分组卷积模拟 P 和 H/W 的局部交互
        self.conv_group = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            # AdaptiveLpShrinkage(channels=out_channels, init_p=0.5),
        )

        # 2. 门控机制 (Gating Mechanism)
        # 用 1x1 卷积生成门控信号
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 3. 融合和降维 (如果 in_channels != out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """ x: (B, C, H, W) 这里的 C 是 U-Net 的隐藏特征维度 """

        # 1. 局部特征提取
        local_feat = self.conv_group(x)  # (B, C, H, W)

        # 2. 门控信号
        gate = self.sigmoid(self.gate_conv(x))  # (B, C, H, W)

        # 3. Gated 激活
        gated_feat = local_feat * gate

        # 4. 最终融合
        out = self.final_conv(gated_feat)

        return out


class GLSSA_UBranch(nn.Module):
    def __init__(self, in_channels, out_channels_r, C=128):
        super().__init__()
        self.r = out_channels_r

        # --- 0. 初始投影 (P -> C) ---
        self.proj_in = ConvBlock(in_channels, C)
        self.initial_attention = ChannelSqueezeBlock(C, reduction=8)  # 初始注意力


        # --- Encoder (Downsampling) ---
        self.enc1 = ConvBlock(C, C)
        self.pool1 = nn.MaxPool2d(2)  # H/2, W/2

        self.enc2 = ConvBlock(C, C * 2)
        self.pool2 = nn.MaxPool2d(2)  # H/4, W/4

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(

            ConvBlock(C * 2, C*4),

            GatedConvBlock(C * 4, C * 2),


        )



        # --- Decoder (Upsampling) ---
        # ConvTranspose2d 用于精确上采样和通道处理
        self.up2 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(C * 2, C)  # C*2 from concatenation (u2 + e1)

        self.up1 = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(C * 2, C)  # C*2 from concatenation (u1 + x0)

        # --- Final Projection ---
        self.final_attention = ChannelSqueezeBlock(C, reduction=8)  # 最终注意力
        self.final_proj = nn.Conv2d(C, self.r, kernel_size=1)



    def forward(self, Y):
        """ Y: (B, P, H, W) -> HSI Input """

        # 0. Initial Projection + Attention
        x = self.proj_in(Y)
        x0 = self.initial_attention(x)  # 初始特征，也是第一个 Skip Connection

        # --- Encoder ---
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)  # 第二个 Skip Connection

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # --- Bottleneck ---
        b = self.bottleneck(p2)
        # b = p2

        # --- Decoder ---
        # 1. Level 2
        u2 = self.up2(b)
        # 检查并调整 u2 尺寸以匹配 e1 的尺寸
        if u2.shape[-2:] != e1.shape[-2:]:
            # 使用插值确保 u2 的 H 和 W 与 e1 完全一致
            u2 = F.interpolate(u2, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        x2 = torch.cat([u2, e1], dim=1)  # 跳跃连接
        d2 = self.dec2(x2)

        # 2. Level 1
        u1 = self.up1(d2)
        # 检查并调整 u1 尺寸以匹配 x0 的尺寸
        if u1.shape[-2:] != x0.shape[-2:]:
            # 使用插值确保 u1 的 H 和 W 与 x0 完全一致
            u1 = F.interpolate(u1, size=x0.shape[-2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([u1, x0], dim=1)  # 跳跃连接
        d1 = self.dec1(x1)

        # --- Final Projection ---
        d1 = self.final_attention(d1)
        U = self.final_proj(d1)

        # Abundance Constraints (Non-negativity)
        U = F.relu(U)
        # U = F.softmax(U, dim=1)


        return U

class SELayer1D(nn.Module):
    """
    一维通道注意力模块 (Squeeze-and-Excitation)
    用于加权 (B, C, L) 形状的特征图中的 C 维度。
    """
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        # Squeeze: 全局平均池化 (B, C, L) -> (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Excitation: 全连接层预测权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # 1. Squeeze: (B, C, 1)
        y = self.avg_pool(x).view(b, c)
        # 2. Excitation: (B, C, 1)
        y = self.fc(y).view(b, c, 1)
        # 3. Scale
        return x * y.expand_as(x)






class HybridPurityNet(nn.Module):
    def __init__(self, p, r, hidden_dim=32):
        super().__init__()
        # 轻量卷积：捕捉局部空间上下文
        self.conv_branch = nn.Sequential(
            nn.Conv2d(p, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, r, kernel_size=1)
        )
        # 端元相似度：提供物理先验
        self.endmembers = nn.Parameter(torch.randn(r, p) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, P, H, W = x.shape
        # 卷积分支
        logits_conv = self.conv_branch(x)                  # (B, R, H, W)
        # 相似度分支
        x_flat = x.reshape(B, P, -1)
        x_norm = F.normalize(x_flat, dim=1)
        e_norm = F.normalize(self.endmembers, dim=1)
        sim = torch.einsum('rp,bpn->brn', e_norm, x_norm)
        logits_sim = sim.reshape(B, -1, H, W)              # (B, R, H, W)
        # 融合
        return logits_conv + self.alpha * logits_sim





class F_PG_Spectral_Branch(nn.Module):
    """
    频域正则化物理引导光谱分支 (Frequency-Regularized PG-PPB)
    结合了空间纯像元锚定与 FFT 频域滤波去噪
    """

    def __init__(self, p, r, hidden_dim=64, reduction=4):
        super().__init__()
        self.p = p
        self.r = r

        # ==========================================
        # 1. 空间纯像元定位网 (PG 核心)
        # ==========================================
        # self.purity_net = HybridPurityNet(p, r, hidden_dim=hidden_dim)
        self.purity_net = nn.Sequential(
            nn.Conv2d(p, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, r, kernel_size=3, padding=1),
        )


        # ==========================================
        # 2. 频域降噪微调网 (Fourier 核心)
        # ==========================================
        self.pre_mlp = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        )

        self.freq_len = p // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(hidden_dim, self.freq_len, 2, dtype=torch.float32) * 0.02
        )

        self.channel_attn = SELayer1D(channel=hidden_dim, reduction=reduction)

        self.post_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, 1),  # 输出 1 个通道的残差
            nn.Tanh()  # 🌟 物理枷锁：将残差幅度限制在 [-1, 1]
        )

        # 限制网络修改光谱的幅度，防止频域过度平滑导致畸变
        self.correction_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, Y):
        B, P, H, W = Y.shape

        # ==========================================
        # 步骤 1：空间提取线性基底 V_base  Spatial Endmember Guidance（空域端元引导）
        # ==========================================
        purity_logits = self.purity_net(Y)
        purity_logits_flat = purity_logits.reshape(B, self.r, -1)
        attn_weights = F.softmax(purity_logits_flat, dim=-1)

        Y_flat = Y.reshape(B, P, -1).transpose(1, 2)
        # V_base: (B, R, P)
        V_base = torch.bmm(attn_weights, Y_flat)

        # ==========================================
        # 步骤 2：利用 Fourier 与 Attention 提取频域降噪残差
        # ==========================================
        # 将 R 条光谱视为 R 个独立的 Batch，压入进行 1D 处理: (B*R, 1, P)
        x_seq = V_base.reshape(B * self.r, 1, P)

        # 预处理升维: (B*R, hidden_dim, P)
        x_emb = self.pre_mlp(x_seq)

        # FFT 变换: (B*R, hidden_dim, freq_len)
        x_fft = torch.fft.rfft(x_emb, dim=-1, norm='ortho')

        # 构造复数权重并滤波
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight

        # 逆变换回时域: (B*R, hidden_dim, P)
        x_filtered = torch.fft.irfft(x_fft, n=P, dim=-1, norm='ortho')

        # 通道注意力精炼重要特征
        x_refined = self.channel_attn(x_filtered)

        # 降维回单波段，并通过 Tanh 限制范围: (B*R, 1, P)
        delta_seq = self.post_mlp(x_refined)

        # 恢复形状: (B, R, P)
        delta = delta_seq.reshape(B, self.r, P)

        # ==========================================
        # 步骤 3：受界物理补偿与约束输出
        # ==========================================
        # 结合基底与受界残差
        V_out = V_base + delta * self.correction_scale

        # 转置为目标形状: (B, P, R)
        V = V_out.permute(0, 2, 1)

        # 严格非负约束，去掉了 L2 归一化以保留反照率
        V = F.relu(V)

        return V





class Ablation_Spectral_Branch_NoPG(nn.Module):
    """
    消融实验版本：移除空间纯度引导 (PG)，仅保留频域滤波
    使用全局可学习参数作为初始端元基底 V_base，切断对输入图像空间分布的依赖。
    """

    def __init__(self, p, r, hidden_dim=64, reduction=4):
        super().__init__()
        self.p = p
        self.r = r

        # 1. 移除空间纯像元定位网 (purity_net)
        # 替换为：直接学习一个全局的、空间无感的初始端元矩阵
        self.V_base = nn.Parameter(torch.rand(1, r, p))
        nn.init.xavier_uniform_(self.V_base)  # 初始化分布

        # 2. 频域降噪微调网 (Fourier 核心保持完全一致)
        self.pre_mlp = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        )

        self.freq_len = p // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(hidden_dim, self.freq_len, 2, dtype=torch.float32) * 0.02
        )

        self.channel_attn = SELayer1D(channel=hidden_dim, reduction=reduction)

        self.post_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Tanh()
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, Y):
        B, P, H, W = Y.shape

        # 步骤 1：直接使用可学习的全局基底，无视图像的空间纯度特征
        # 扩展到当前 Batch 大小: (B, R, P)
        V_base = self.V_base.expand(B, -1, -1)

        # 步骤 2：利用 Fourier 与 Attention 提取频域降噪残差 (维持原样)
        x_seq = V_base.reshape(B * self.r, 1, P)
        x_emb = self.pre_mlp(x_seq)

        x_fft = torch.fft.rfft(x_emb, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x_filtered = torch.fft.irfft(x_fft, n=P, dim=-1, norm='ortho')

        x_refined = self.channel_attn(x_filtered)
        delta_seq = self.post_mlp(x_refined)
        delta = delta_seq.reshape(B, self.r, P)

        # 步骤 3：结合基底与受界残差
        V_out = V_base + delta * self.correction_scale
        V = V_out.permute(0, 2, 1)
        V = F.relu(V)

        return V


class Model(nn.Module):
    def __init__(self, p, r, base_dim=128):
        super().__init__()

        self.Abundance_Module = GLSSA_UBranch(in_channels=p, out_channels_r=r, C=base_dim)
        self.SpectralBasis_Module = F_PG_Spectral_Branch(p=p, r=r, hidden_dim=base_dim)


    def forward(self, y):

        U = self.Abundance_Module(y)
        V = self.SpectralBasis_Module(y)
        Y_hat = torch.einsum('bpr,brhw->bphw', V, U)

        return {'U': U, 'V': V, 'Y_hat': Y_hat}


class Model_ablation(nn.Module):
    def __init__(self, p, r, base_dim=128):
        super().__init__()

        self.Abundance_Module = GLSSA_UBranch(in_channels=p, out_channels_r=r, C=base_dim)

        self.SpectralBasis_Module = Ablation_Spectral_Branch_NoPG(p=p, r=r, hidden_dim=base_dim)
        # self.SpectralBasis_Module = F_PG_Spectral_Branch(p=p, r=r, hidden_dim=base_dim)


    def forward(self, y):

        U = self.Abundance_Module(y)
        V = self.SpectralBasis_Module(y)
        Y_hat = torch.einsum('bpr,brhw->bphw', V, U)

        return {'U': U, 'V': V, 'Y_hat': Y_hat}

