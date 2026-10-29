from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam

import matplotlib.pyplot as plt

from visualization import painting_rl


def _to_numpy_image(X):
    """内部工具：把输入转换为 numpy array，shape -> (p, H, W), dtype float32, values任意范围"""
    X_np = (
        X.cpu()  # 第一步：将 Tensor 从 GPU 复制到 CPU
        .squeeze(0)  # 第二步：去除批次维度（B=1），形状变为 (C, H, W)
        .permute(1, 2, 0)  # 第三步：调整维度为 (H, W, C)
        .numpy()  # 第四步：转为 NumPy 数组
    )
    return X_np.astype(np.float32)


def plot_metric_vs_loss(metric_log, loss_log, metric_name="PSNR", smooth_sigma=2,
                        metric_color="#1f77b4", loss_color="#d62728",
                        metric_label_left=None, loss_label_right=None,
                        show_samples=None):
    """
    绘制指标(PSNR/SSIM/SAM)与Loss双坐标曲线。
    参数：
        metric_log : list/array，记录每次迭代的指标值
        loss_log   : list/array，记录每次迭代的loss或RelErr
        metric_name: str 指标名称 ("PSNR", "SSIM", "SAM")
        smooth_sigma: float，高斯平滑sigma值
        metric_color: str 左轴曲线颜色
        loss_color: str 右轴曲线颜色
        metric_label_left: 左轴标签
        loss_label_right : 右轴标签
        show_samples: list[np.ndarray]，中间可视化小图（RGB图），自动排列
    """
    iters = np.arange(1, len(metric_log)+1)
    # 平滑
    metric_smooth = gaussian_filter1d(metric_log, sigma=smooth_sigma)
    loss_smooth = gaussian_filter1d(loss_log, sigma=smooth_sigma)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 左轴：性能指标
    ax1.plot(iters, metric_smooth, color=metric_color, lw=2)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel(metric_label_left or metric_name, color=metric_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=metric_color)

    # 右轴：Loss
    ax2 = ax1.twinx()
    ax2.plot(iters, loss_smooth, color=loss_color, lw=1.2, alpha=0.8)
    ax2.set_ylabel(loss_label_right or "RelErr", color=loss_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=loss_color)

    # 美化
    ax1.grid(alpha=0.3)
    plt.title(f"{metric_name} and RelErr vs Iteration", fontsize=14, weight='bold')

    # 嵌入中间结果图（如论文样式）
    if show_samples is not None and len(show_samples) > 0:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        step = len(iters) // (len(show_samples) + 1)
        for i, img in enumerate(show_samples):
            it = (i+1)*step
            zoom = 0.25 if img.shape[0] > 100 else 0.5
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (it, metric_smooth[it-1]), frameon=True,
                                pad=0.3, boxcoords="data", bboxprops=dict(edgecolor="#d27d47", lw=1.5))
            ax1.add_artist(ab)

    plt.tight_layout()
    plt.show()











def enhanced_spectral_tv_loss(Y_hat):
    """
    增强的光谱TV损失
    """
    # 一阶差分
    diff1 = torch.abs(Y_hat[:, 1:, :, :] - Y_hat[:, :-1, :, :])
    spectral_tv_1 = torch.mean(diff1)

    # # 二阶差分
    # diff2 = torch.abs(diff1[:, 1:, :, :] - diff1[:, :-1, :, :])
    # spectral_tv_2 = torch.mean(diff2)
    #
    # # 三阶差分
    # diff3 = torch.abs(diff2[:, 1:, :, :] - diff2[:, :-1, :, :])
    # spectral_tv_3 = torch.mean(diff3)
    #
    # # return spectral_tv_1 + 0.5 * spectral_tv_2 + 0.1 * spectral_tv_3
    return spectral_tv_1







def plot_early_stopping_analysis(psnr_log, ae_loss_log, actual_stop_iter=None, save_path=None):
    """
    PSNR 与 AE Loss 双轴对比图。
    3:2 画幅；标题居中；图例置于整张 figure 右上角；
    显式设置 Times New Roman；用 subplots_adjust 保证左侧 ylabel 不被裁切。
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ================= 字体：Times New Roman =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    iters = np.arange(1, len(psnr_log) + 1)

    fig, ax1 = plt.subplots(figsize=(18, 12))

    # ================= 左轴：PSNR =================
    color1 = 'tab:blue'
    ax1.set_xlabel('Iteration', fontsize=34, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=34, fontweight='bold')
    ax1.plot(iters, psnr_log, color=color1, label='PSNR', linewidth=5.5)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=34)
    ax1.tick_params(axis='x', labelsize=34)

    best_psnr_iter = np.argmax(psnr_log)
    best_psnr_val = psnr_log[best_psnr_iter]
    ax1.plot(best_psnr_iter + 1, best_psnr_val, 'bo', markersize=20)

    # ================= 右轴：AE Loss =================
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('AE Loss (MSE)', color=color2, fontsize=34, fontweight='bold')
    ax2.plot(iters, ae_loss_log, color=color2, label='AE Loss',
             linewidth=5.5, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=34)

    # ================= 早停位置 =================
    if actual_stop_iter is not None and actual_stop_iter > 0:
        ax2.axvline(x=actual_stop_iter, color='gray',
                    linestyle=':', linewidth=5.5)
        offset = max(10, int(0.02 * len(iters)))
        ax2.text(actual_stop_iter + offset, np.mean(ae_loss_log),
                 f'Stop Triggered\nat Iter {actual_stop_iter}',
                 color='dimgray', fontsize=26, fontweight='bold')

    # ================= 标题：居中 =================
    ax1.set_title('Early Stopping Analysis',
                  fontsize=42, fontweight='bold', pad=36)

    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=2.0)

    # ================= 图例：整张 figure 右上角 =================
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    fig.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        ncol=1,
        fontsize=26,
        frameon=True,
        framealpha=0.85,
        edgecolor='black',
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.8,
        handletextpad=0.6
    )

    # ================= 精确控制四周边距，替代 tight_layout =================
    # left/right 给两侧 ylabel 和刻度留足空间；top 给标题与图例；bottom 给 xlabel
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.11, top=0.86)

    # ================= 保存 =================
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if not os.path.splitext(save_path)[1]:
            save_path = save_path + '.pdf'
        # pad_inches 从 0.05 放宽到 0.2，避免边缘内容被裁
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        print(f"图像已保存至: {save_path}")

    plt.show()
    plt.close(fig)

# ========== 指标 ==========
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(target.max() / torch.sqrt(mse))

def spectral_angle_mapper(pred, target):
    # pred, target: (B, P, H, W)
    B, P, H, W = pred.shape
    pred_flat = pred.reshape(B, P, -1)
    target_flat = target.reshape(B, P, -1)
    dot = (pred_flat * target_flat).sum(dim=1)
    denom = (pred_flat.norm(dim=1) * target_flat.norm(dim=1) + 1e-8)
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean() * 180 / np.pi

def ssim(pred, target, window_size=11):
    """
    Compute SSIM for hyperspectral image (band-wise average).
    pred, target: [B, P, H, W]
    """
    B, P, H, W = pred.shape
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_x2
    sigma_y2 = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()

















# ===================== 先把 AE 网络 + 早停类放到顶部（直接用）=====================


class SS_AE_Validator(nn.Module):
    """
    空谱解耦自编码器 (Spatial-Spectral AE Validator)

    """

    def __init__(self, bands, hidden_dim=16):
        super().__init__()
        self.bands = bands

        # ==========================================
        # 编码器：空谱解耦压缩
        # ==========================================
        # 1. 空间感知：Depthwise Conv (独立处理每个波段，捕捉空间伪影)
        self.spatial_enc = nn.Conv2d(bands, bands, kernel_size=3, stride=2, padding=1, groups=bands)

        # 2. 光谱压缩：Pointwise Conv (跨波段融合，捕捉光谱畸变)
        self.spectral_enc = nn.Conv2d(bands, hidden_dim, kernel_size=1, stride=1)

        self.act = nn.GELU()

        # ==========================================
        # 解码器：对称还原
        # ==========================================
        # 1. 空间还原：保持组卷积特性
        self.spatial_dec = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1,
                                              output_padding=1, groups=hidden_dim)

        # 2. 光谱还原：1x1卷积恢复至原始波段数
        self.spectral_dec = nn.Conv2d(hidden_dim, bands, kernel_size=1, stride=1)

    def forward(self, x):
        """
        直接对完整图像进行前向传播，无任何掩蔽操作
        """
        # 1. 解耦编码
        h = self.spatial_enc(x)
        h = self.act(self.spectral_enc(h))

        # 2. 解耦解码
        h = self.act(self.spatial_dec(h))
        out = self.spectral_dec(h)

        return out

class SelfValidationEarlyStop:
    """
    创新点 : Trend-Aware Delayed Stopping (TADS) 趋势感知与预热保护，免疫前期剧烈震荡
    """

    def __init__(self, bands, patience=50, window_size=100, lr=1e-3,
                 burn_in=300, trend_window=10, min_delta=1e-5):
        """
        :param burn_in: 预热保护期。在前 burn_in 轮内，即使 Loss 上升也绝不触发早停。
        :param trend_window: 趋势计算窗口。比较最近 trend_window 轮与上一个 trend_window 轮的均值。
        :param min_delta: 最小下降阈值。如果下降幅度低于此值，认定为进入平台期或反弹。
        """
        self.patience = patience
        self.window_size = window_size
        self.image_buffer = []

        self.burn_in = burn_in
        self.trend_window = trend_window
        self.min_delta = min_delta
        self.loss_history = []  # 记录验证 Loss 以计算滑动斜率

        # 初始化 AE + 优化器
        self.ae = SS_AE_Validator(bands).cuda()
        self.opt = optim.Adam(self.ae.parameters(), lr=lr)

        self.best_loss = float('inf')
        self.wait = 0
        self.best_iter = 0

    def add_image(self, img_tensor):
        """加入当前重建图像到滑动窗口"""
        self.image_buffer.append(img_tensor.detach())
        if len(self.image_buffer) > self.window_size:
            self.image_buffer.pop(0)

    def update_ae(self):
        """用滑动窗口的图像训练 AE（只走一步梯度）"""
        if len(self.image_buffer) < 5:
            return 0.0

        self.ae.train()
        self.opt.zero_grad()

        # 优化：随机从 buffer 中抽 4 张图组成一个小 batch
        indices = torch.randperm(len(self.image_buffer))[:4]
        batch = torch.stack([self.image_buffer[i] for i in indices]).squeeze(1)  # [batch, P, H, W]





        recon = self.ae(batch)
        _, _, H, W = batch.shape
        recon_cropped = recon[:, :, :H, :W]


        loss = F.l1_loss(batch, recon_cropped)
        loss.backward()
        self.opt.step()

        return loss.item()


    def compute_loss(self, img):
        """计算Loss"""
        self.ae.eval()
        with torch.no_grad():

            recon = self.ae(img)

            _, _, H, W = img.shape
            recon_cropped = recon[:, :, :H, :W]

            return F.l1_loss(img, recon_cropped).item()

    def check_stop(self, current_loss, current_iter):
        """核心早停判断：基于局部趋势斜率与预热保护机制"""
        self.loss_history.append(current_loss)

        # ---------------------------------------------------------
        # 1. 预热保护期 (Burn-in Period)
        # 前期模型重构极其不稳定，不累加 patience，仅默默记录最低点
        # ---------------------------------------------------------
        if current_iter <= self.burn_in:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_iter = current_iter
            return False

        # 确保有足够的数据来计算前后两个窗口的均值
        if len(self.loss_history) < 2 * self.trend_window:
            return False

        # ---------------------------------------------------------
        # 2. 局部趋势斜率判断 (Local Trend Slope)
        # ---------------------------------------------------------
        # 计算最近一个窗口的均值 (Current Trend)
        recent_avg = sum(self.loss_history[-self.trend_window:]) / self.trend_window
        # 计算上一个窗口的均值 (Past Trend)
        past_avg = sum(self.loss_history[-2 * self.trend_window: -self.trend_window]) / self.trend_window

        # 如果当前均值 >= 过去均值 (考虑了极小抖动 min_delta) -> 曲线斜率变平或上扬
        if recent_avg >= past_avg - self.min_delta:
            self.wait += 1
        else:
            # 曲线仍在健康下降 -> 重置耐心值
            self.wait = 0
            # 同步更新全局最佳记录（用于最后输出汇报）
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_iter = current_iter

        # 内存管理：只保留计算趋势所需的历史数据，防止列表无限增长
        if len(self.loss_history) > 3 * self.trend_window:
            self.loss_history.pop(0)

        # ---------------------------------------------------------
        # 3. 触发早停
        # ---------------------------------------------------------
        if self.wait >= self.patience:
            print(f"\n[Early Stopping Triggered] At iteration {current_iter}.")
            print(f"Optimal reconstruction point found around iteration {self.best_iter}.")
            return True

        return False




# ===================== 你的原训练函数（已替换为论文早停）=====================
def train_dsdp2(model, Y_noisy, Y_clean, alpha=0.1, mask_ratio=0.5, lr=1e-3,
                gamma=0.9, max_iter=2000,save=None):
    """
    model: DSDPModel 实例
    Y: 输入噪声高光谱图像 [B, P, H, W]
    """
    # ✅ 替换成：Self-Validation 自验证早停（自动获取高光谱波段数）
    bands = Y_noisy.shape[1]
    es = SelfValidationEarlyStop(bands=bands, patience=50, window_size=30,burn_in=200)

    # 1. 优化器
    optimizer = Adam(model.parameters(), lr=lr)


    out_prev = Y_noisy.clone()
    Y_tgt = Y_noisy.clone()
    loss_log, psnr_log, ssim_log, sam_log = [], [], [], []
    ae_loss_log = []
    max_psnr = 0
    max_ssim = 0
    min_sam = 0
    iter = 0
    f = True
    stop_at_iter = 0
    for t in range(max_iter):

        optimizer.zero_grad()


        # Noise2Void 掩码
        mask = (torch.rand_like(Y_noisy) < mask_ratio).float()
        Y_masked = Y_noisy * (1 - mask)



        model.train()

        out = model(Y_masked)
        Y_hat = out['Y_hat']
        U, V = out['U'], out['V']



        loss_mask = F.l1_loss(Y_hat[mask.bool()], Y_tgt[mask.bool()])
        # loss_mask = F.l1_loss(Y_hat, Y_tgt)
        image_spectral_tv = enhanced_spectral_tv_loss(Y_hat)

        # 总损失
        loss = loss_mask + alpha * image_spectral_tv

        loss.backward()
        optimizer.step()

        # 平滑输出
        out_smooth = gamma * out_prev + (1 - gamma) * Y_hat.detach()
        out_prev = out_smooth

        # ====== 核心：Self-Validation 早停逻辑（每一步都执行）======
        es.add_image(out_smooth)  # 把当前最优重建图存入滑动窗口
        es.update_ae()  # 训练 AE（在线学习）
        ae_loss = es.compute_loss(out_smooth)  # 计算 AE 重构损失

        # ====== 指标计算 ======
        cur_psnr = psnr(out_smooth, Y_clean).item()
        cur_sam = spectral_angle_mapper(out_smooth, Y_clean).item()
        cur_ssim = ssim(out_smooth, Y_clean).item()

        if cur_psnr > max_psnr:
            max_psnr = cur_psnr
            max_ssim = cur_ssim
            min_sam = cur_sam
            iter = t + 1

        loss_log.append(loss.item())
        psnr_log.append(cur_psnr)
        sam_log.append(cur_sam)
        ssim_log.append(cur_ssim)
        ae_loss_log.append(ae_loss)

        # if (t + 1) % 100 == 0:
        #     Y_noisy1 = _to_numpy_image(Y_noisy)
        #     out_prev1 = _to_numpy_image(out_prev)
        #
        #     # painting_rl(Y_noisy1, out_prev1, [165, 96, 6], name=f'{t + 1}', path='visualization/gf_330/gf(329,201,2)')#E0
        #     painting_rl(Y_noisy1, out_prev1, [245, 195, 75], name=f'{t + 1}', path='visualization/gf_330/gf(329,201,2)')#GF
        #     # painting_rl(Y_noisy1, out_prev1, [154, 95, 42], name=f'{t + 1}', path='visualization/zy(89, 40, 1)')#GF-shanghai
        #     # painting_rl(Y_noisy1, out_prev1, [89, 40, 1], name=f'{t + 1}',path='visualization/gf_330/gf(329,201,2)')  # ZY
        #     # painting_rl(Y_noisy1, out_prev1, [0, 103, 187], name=f'{t + 1}',path='visualization/zy(89, 40, 1)')  # Urban

        # ====== 打印 ======
        if (t + 1) % 100 == 0:
            print(f"[Iter {t + 1:4d}] "
                  f"Loss={loss.item():.6f} | PSNR={cur_psnr:.2f} | "
                  f"SSIM={cur_ssim:.3f} | SAM={cur_sam:.3f}° | "
                  f"AE Loss={ae_loss:.6f} | Wait={es.wait}")

        # ✅ 论文早停触发（替换你原来的 rel_err 早停）
        if  f:
         if es.check_stop(ae_loss, t):
            print(f"\n✅ [Self-Validation 早停] 触发于迭代 {t + 1}")
            print(f"✅ 最佳迭代: {es.best_iter + 1} | 最佳AE损失: {es.best_loss:.6f}")
            stop_at_iter = t+1
            f = False
            # break



    plot_early_stopping_analysis(psnr_log, ae_loss_log, actual_stop_iter=stop_at_iter,save_path=save)
    print(f"[Iter {iter :4d}] "
          f"MaxPSNR={max_psnr:.2f} | MaxSSIM={max_ssim:.3f} | MinSAM={min_sam:.3f}° ")

    return out_prev, U, V, loss_log, psnr_log, ssim_log, sam_log








if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from model import Model  # 你的模型文件
    import scipy.io as sio
    import time
    import h5py
    import os
    import torch

    # 结果保存根目录
    save_root = 'comparative_experiment'
    os.makedirs(save_root, exist_ok=True)

    # list = ['../data/noisy/WDC_a.mat','../data/noisy/WDC_b.mat','../data/noisy/WDC_c.mat','../data/noisy/WDC_d.mat','../data/noisy/WDC_e.mat','../data/noisy/WDC_f.mat']
    list = ['../data/noisy/Houston_a.mat',  '../data/noisy/Houston_c.mat', '../data/noisy/Houston_f.mat']
    # save = [r'ablation\early_stop\Houston_a_stop.pdf',r'ablation\early_stop\Houston_b_stop.pdf',r'ablation\early_stop\Houston_c_stop.pdf',r'ablation\early_stop\Houston_d_stop.pdf',r'ablation\early_stop\Houston_e_stop.pdf',r'ablation\early_stop\Houston_f_stop.pdf']
    save = [r'ablation\early_stop\WDC_a_stop.pdf', r'ablation\early_stop\WDC_c_stop.pdf',r'ablation\early_stop\WDC_f_stop.pdf']

    for i in range(1):
        # ========== 1. 加载数据 ==========
        # mat_path = r'../data/real/GF-5.mat'
        # mat_path = r'../data/real/ZY1E02D.mat'
        mat_path = r'../data/noisy/Houston_f.mat'
        # mat_path = list[i]
        try:
            mat = sio.loadmat(mat_path)
        except NotImplementedError:
            mat = {}
            with h5py.File(mat_path, 'r') as f:
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        mat[k] = np.array(f[k]).T

        print(mat.keys())
        Y_clean = mat['gt']
        Y_noisy = mat['noisy']
        print('已加载文件 ' + mat_path)
        print(Y_clean.shape)
        print(Y_noisy.shape)

        # ========== 2. 转成 tensor，维度顺序 (1, P, H, W) ==========
        Y_noisy = torch.from_numpy(Y_noisy.astype(np.float32)).unsqueeze(0).permute(0, 3, 1, 2)
        Y_clean = torch.from_numpy(Y_clean.astype(np.float32)).unsqueeze(0).permute(0, 3, 1, 2)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Y_noisy = Y_noisy.to(device)
        Y_clean = Y_clean.to(device)
        C, P, H, W = Y_clean.shape

        # ========== 3. 模型与去噪 ==========
        model = Model(p=P, r=24, base_dim=256).cuda()

        start = time.time()
        Y_denoised, U, V, loss_log, psnr_log, ssim_log, sam_log = train_dsdp2(
            model, Y_noisy, Y_clean, alpha=0.1, lr=1e-3, mask_ratio=0.5,max_iter=1500,gamma=0.8,save=r'ablation\early_stop\WDC_f_stop.pdf'
        )
        end = time.time()

        elapsed_time = end - start
        print(f"执行时间: {elapsed_time:.4f} 秒")

        # ========== 4. 评估指标 ==========
        psnr_before = psnr(Y_noisy, Y_clean)
        ssim_before = ssim(Y_noisy, Y_clean)
        sam_before = spectral_angle_mapper(Y_noisy, Y_clean)

        psnr_after = psnr(Y_denoised, Y_clean)
        ssim_after = ssim(Y_denoised, Y_clean)
        sam_after = spectral_angle_mapper(Y_denoised, Y_clean)

        print(f"PSNR: {psnr_before:.2f} → {psnr_after:.2f}")
        print(f"SSIM: {ssim_before:.3f} → {ssim_after:.3f}")
        print(f"SAM : {sam_before:.3f}° → {sam_after:.3f}°")

        # ========== 5. 保存结果 ==========

        # noisy_np    = _to_numpy_image(Y_noisy)
        # gt_np       = _to_numpy_image(Y_clean)
        # denoised_np = _to_numpy_image(Y_denoised)
        #
        # # 用原文件名（不含扩展名）作为子目录名
        # file_stem = os.path.splitext(os.path.basename(mat_path))[0]  # 如 Houston_a
        # save_dir = os.path.join(save_root, file_stem)
        # os.makedirs(save_dir, exist_ok=True)
        #
        # save_path = os.path.join(save_dir, 'Ours.mat')
        # sio.savemat(save_path, {
        #     'denoised': denoised_np,
        #     'time': elapsed_time
        # }, do_compression=True)
        #
        # print(f"结果已保存到: {save_path}")


