"""
STAFF v3.0: Spatio-Temporal Adaptive Feature Fusion Module
时空自适应特征融合模块

核心特性:
    - 限幅空间滤波 (γ ∈ [0.1, 0.5])
    - 向量几何偏差度量 (方向 + 幅度 + 空间异质)
    - 固定曲率双曲融合 (c = 1.0)
    - 无需边特征，仅依赖节点特征和邻接矩阵

作者: STAFF Team
版本: 3.0 (工程优化版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class STAFFModule(nn.Module):
    """
    STAFF 模块主类
    
    输入:
        X_h: 历史记忆图 [B, N, D] (TCN/GRU编码后)
        X_c: 当前观测图 [B, N, D]
        A:   邻接矩阵 [N, N]
    
    输出:
        X_out: 融合特征 [B, N, D]
        alpha: 门控权重 [B, N, 1]
    """
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int = 64,
        gamma_min: float = 0.1,
        gamma_max: float = 0.5,
        gamma_init: float = 0.3,
        curvature: float = 1.0,
        eps: float = 1e-5
    ):
        """
        初始化STAFF模块
        
        参数:
            in_dim:      输入特征维度 D
            hidden_dim:  门控MLP隐藏层维度 (默认64)
            gamma_min:   空间滤波系数下限 (默认0.1)
            gamma_max:   空间滤波系数上限 (默认0.5)
            gamma_init:  空间滤波系数初始值 (默认0.3)
            curvature:   双曲空间曲率 (默认1.0，固定)
            eps:         数值稳定性常数 (默认1e-5)
        """
        super(STAFFModule, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = eps
        
        # ========== 1. 空间滤波系数 (限幅可学习) ==========
        # 使用logit初始化，保证初始gamma在期望值
        gamma_logit = torch.log(torch.tensor(gamma_init) / (1 - gamma_init))
        self.gamma_hat = nn.Parameter(gamma_logit)
        
        # ========== 2. 双曲曲率 (固定) ==========
        # 注册为buffer，不参与梯度更新
        self.register_buffer('curvature', torch.tensor(curvature))
        
        # ========== 3. 门控网络 (3维输入 -> 1维输出) ==========
        # 输入：[F_dir, F_mag, F_space]
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # ========== 4. 输出投影层 ==========
        self.out_proj = nn.Linear(in_dim, in_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def _get_norm_adj(self, A: torch.Tensor) -> torch.Tensor:
        """
        计算对称归一化邻接矩阵
        
        公式: A_norm = D^(-1/2) A D^(-1/2)
        
        参数:
            A: 邻接矩阵 [N, N]
        
        返回:
            A_norm: 归一化邻接矩阵 [N, N]
        """
        # 计算度矩阵
        D = torch.sum(A, dim=1)  # [N]
        
        # 计算 D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + self.eps))  # [N, N]
        
        # 对称归一化
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # [N, N]
        
        return A_norm
    
    def _spatial_filter(
        self, 
        X: torch.Tensor, 
        A_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        图拉普拉斯空间滤波 (带限幅)
        
        公式: X̂ = (1 - γ)X + γ * A_norm @ X
        
        参数:
            X:      节点特征 [B, N, D]
            A_norm: 归一化邻接矩阵 [N, N]
        
        返回:
            X_filt: 滤波后特征 [B, N, D]
        """
        # 计算有效γ (Sigmoid + Clamp)
        gamma_raw = torch.sigmoid(self.gamma_hat)
        gamma = torch.clamp(gamma_raw, min=self.gamma_min, max=self.gamma_max)
        
        # 邻居聚合
        neighbor_agg = torch.matmul(A_norm, X)  # [B, N, D]
        
        # 加权融合
        X_filt = (1 - gamma) * X + gamma * neighbor_agg
        
        return X_filt
    
    def _calc_directional_shift(
        self, 
        X_h: torch.Tensor, 
        X_c: torch.Tensor
    ) -> torch.Tensor:
        """
        方向性偏移 (余弦距离)
        
        公式: F_dir = 1 - cosine(X_h, X_c)
        
        参数:
            X_h: 历史特征 [B, N, D]
            X_c: 当前特征 [B, N, D]
        
        返回:
            F_dir: 方向偏移 [B, N, 1]
        """
        # 计算L2范数
        norm_h = torch.norm(X_h, p=2, dim=-1, keepdim=True) + self.eps  # [B, N, 1]
        norm_c = torch.norm(X_c, p=2, dim=-1, keepdim=True) + self.eps  # [B, N, 1]
        
        # 计算内积
        dot = torch.sum(X_h * X_c, dim=-1, keepdim=True)  # [B, N, 1]
        
        # 计算余弦相似度
        cos_sim = dot / (norm_h * norm_c)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 数值稳定
        
        # 余弦距离
        F_dir = 1.0 - cos_sim  # [B, N, 1]
        
        return F_dir
    
    def _calc_magnitude_shock(
        self, 
        X_h: torch.Tensor, 
        X_c: torch.Tensor
    ) -> torch.Tensor:
        """
        幅度性冲击 (相对误差)
        
        公式: F_mag = |‖X_c‖ - ‖X_h‖| / ‖X_h‖
        
        参数:
            X_h: 历史特征 [B, N, D]
            X_c: 当前特征 [B, N, D]
        
        返回:
            F_mag: 幅度冲击 [B, N, 1]
        """
        # 计算L2范数
        norm_h = torch.norm(X_h, p=2, dim=-1, keepdim=True) + self.eps  # [B, N, 1]
        norm_c = torch.norm(X_c, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        
        # 相对误差
        F_mag = torch.abs(norm_c - norm_h) / norm_h  # [B, N, 1]
        
        return F_mag
    
    def _calc_spatial_var(
        self, 
        X: torch.Tensor, 
        A_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        空间异质性 (邻居方差)
        
        公式: F_space = A_norm @ (X - neighbor_mean)^2
        
        参数:
            X:      节点特征 [B, N, D]
            A_norm: 归一化邻接矩阵 [N, N]
        
        返回:
            F_space: 空间异质 [B, N, 1]
        """
        # 计算邻居均值
        neighbor_mean = torch.matmul(A_norm, X)  # [B, N, D]
        
        # 计算偏差
        diff = X - neighbor_mean  # [B, N, D]
        
        # 加权方差 (沿特征维度平均)
        variance = torch.matmul(A_norm, diff ** 2)  # [B, N, D]
        F_space = torch.mean(variance, dim=-1, keepdim=True)  # [B, N, 1]
        
        return F_space
    
    def _exp_map_0(self, x: torch.Tensor) -> torch.Tensor:
        """
        欧氏空间 → 双曲空间 (指数映射)
        
        公式: z = tanh(√c‖x‖) * x / (√c‖x‖)
        
        参数:
            x: 欧氏特征 [B, N, D]
        
        返回:
            z: 双曲特征 [B, N, D]
        """
        c = self.curvature
        
        # 计算范数
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        
        # 指数映射
        sqrt_c_norm = torch.sqrt(c) * norm
        tanh_val = torch.tanh(sqrt_c_norm)
        z = tanh_val * x / (sqrt_c_norm + self.eps)
        
        return z
    
    def _log_map_0(self, z: torch.Tensor) -> torch.Tensor:
        """
        双曲空间 → 欧氏空间 (对数映射)
        
        公式: x = arctanh(√c‖z‖) * z / (√c‖z‖)
        
        参数:
            z: 双曲特征 [B, N, D]
        
        返回:
            x: 欧氏特征 [B, N, D]
        """
        c = self.curvature
        
        # 计算范数
        norm = torch.norm(z, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        
        # 对数映射
        sqrt_c_norm = torch.sqrt(c) * norm
        arctanh_arg = torch.clamp(sqrt_c_norm, max=0.999)  # 防止arctanh爆炸
        arctanh_val = torch.arctanh(arctanh_arg)
        x = arctanh_val * z / (sqrt_c_norm + self.eps)
        
        return x
    
    def _mobius_scalar_mul(
        self, 
        r: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Möbius 标量乘法
        
        公式: r ⊗_c z = tanh(r * arctanh(√c‖z‖)) * z / (√c‖z‖)
        
        参数:
            r: 标量权重 [B, N, 1]
            z: 双曲特征 [B, N, D]
        
        返回:
            z_scaled: 缩放后双曲特征 [B, N, D]
        """
        c = self.curvature
        
        # 计算范数
        norm = torch.norm(z, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c_norm = torch.sqrt(c) * norm
        
        # Möbius 标量乘
        arctanh_val = torch.arctanh(torch.clamp(sqrt_c_norm, max=0.999))
        tanh_val = torch.tanh(r * arctanh_val)
        z_scaled = tanh_val * z / (sqrt_c_norm + self.eps)
        
        return z_scaled
    
    def _mobius_add(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Möbius 加法
        
        公式: u ⊕_c v = [(1 + 2c⟨u,v⟩ + c‖v‖²)u + (1 - c‖u‖²)v] / 
                        [1 + 2c⟨u,v⟩ + c²‖u‖²‖v‖²]
        
        参数:
            u: 双曲特征1 [B, N, D]
            v: 双曲特征2 [B, N, D]
        
        返回:
            w: 融合后双曲特征 [B, N, D]
        """
        c = self.curvature
        
        # 计算内积和范数平方
        uv_inner = torch.sum(u * v, dim=-1, keepdim=True)  # [B, N, 1]
        u_norm2 = torch.sum(u ** 2, dim=-1, keepdim=True)  # [B, N, 1]
        v_norm2 = torch.sum(v ** 2, dim=-1, keepdim=True)  # [B, N, 1]
        
        # 分子
        num = (1 + 2 * c * uv_inner + c * v_norm2) * u + \
              (1 - c * u_norm2) * v
        
        # 分母
        den = 1 + 2 * c * uv_inner + c ** 2 * u_norm2 * v_norm2
        
        # Möbius 加法
        w = num / (den + self.eps)
        
        return w
    
    def forward(
        self, 
        X_h: torch.Tensor, 
        X_c: torch.Tensor, 
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            X_h: 历史记忆图 [B, N, D]
            X_c: 当前观测图 [B, N, D]
            A:   邻接矩阵 [N, N]
        
        返回:
            X_out: 融合特征 [B, N, D]
            alpha: 门控权重 [B, N, 1]
        """
        B, N, D = X_h.shape
        
        # 验证输入形状
        assert X_c.shape == (B, N, D), "X_c 形状必须与 X_h 一致"
        assert A.shape == (N, N), "A 形状必须为 [N, N]"
        
        # ========== 步骤 1: 空间滤波 ==========
        A_norm = self._get_norm_adj(A)
        X_h_filt = self._spatial_filter(X_h, A_norm)
        X_c_filt = self._spatial_filter(X_c, A_norm)
        
        # ========== 步骤 2: 门控特征计算 ==========
        F_dir = self._calc_directional_shift(X_h_filt, X_c_filt)    # [B, N, 1]
        F_mag = self._calc_magnitude_shock(X_h_filt, X_c_filt)      # [B, N, 1]
        F_space = self._calc_spatial_var(X_c_filt, A_norm)          # [B, N, 1]
        
        # 特征拼接
        F_gate = torch.cat([F_dir, F_mag, F_space], dim=-1)         # [B, N, 3]
        
        # 生成门控权重
        alpha = self.gate_mlp(F_gate)                               # [B, N, 1]
        
        # ========== 步骤 3: 双曲空间融合 ==========
        # 欧氏→双曲
        z_h = self._exp_map_0(X_h_filt)
        z_c = self._exp_map_0(X_c_filt)
        
        # Möbius 标量乘 (门控作用)
        z_h_weighted = self._mobius_scalar_mul(1 - alpha, z_h)
        z_c_weighted = self._mobius_scalar_mul(alpha, z_c)
        
        # Möbius 加法 (特征融合)
        z_fusion = self._mobius_add(z_h_weighted, z_c_weighted)
        
        # 双曲→欧氏
        X_fusion = self._log_map_0(z_fusion)
        
        # ========== 步骤 4: 输出投影 ==========
        X_out = self.out_proj(X_fusion)
        
        return X_out, alpha
    
    def get_gamma(self) -> float:
        """获取当前有效的空间滤波系数γ"""
        gamma_raw = torch.sigmoid(self.gamma_hat)
        gamma = torch.clamp(gamma_raw, min=self.gamma_min, max=self.gamma_max)
        return gamma.item()
    
    def get_curvature(self) -> float:
        """获取双曲曲率c"""
        return self.curvature.item()


# ============================================================================
# 辅助工具函数
# ============================================================================

def get_common_adj(
    A_h: Optional[torch.Tensor],
    A_c: Optional[torch.Tensor],
    method: str = 'union'
) -> torch.Tensor:
    """
    生成统一邻接矩阵 (用于处理动态图拓扑)
    
    参数:
        A_h:    历史图邻接矩阵 [N, N] (可选)
        A_c:    当前图邻接矩阵 [N, N] (可选)
        method: 合并方法 ['union', 'intersection', 'average', 'history', 'current']
    
    返回:
        A_common: 统一邻接矩阵 [N, N]
    """
    if A_h is None and A_c is None:
        raise ValueError("至少提供一个邻接矩阵")
    
    if A_h is None:
        return A_c
    if A_c is None:
        return A_h
    
    if method == 'union':
        return torch.max(A_h, A_c)
    elif method == 'intersection':
        return torch.min(A_h, A_c)
    elif method == 'average':
        return (A_h + A_c) / 2
    elif method == 'history':
        return A_h
    elif method == 'current':
        return A_c
    else:
        raise ValueError(f"未知方法: {method}")


def create_adj_matrix(
    num_nodes: int, 
    edge_index: torch.Tensor,
    weighted: bool = False,
    edge_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    从边索引创建邻接矩阵
    
    参数:
        num_nodes:   节点数量
        edge_index:  边索引 [2, E]
        weighted:    是否加权
        edge_weights: 边权重 [E] (如果weighted=True)
    
    返回:
        A: 邻接矩阵 [N, N]
    """
    A = torch.zeros(num_nodes, num_nodes)
    
    if weighted and edge_weights is not None:
        A[edge_index[0], edge_index[1]] = edge_weights
    else:
        A[edge_index[0], edge_index[1]] = 1.0
    
    # 对称化
    A = (A + A.T) / 2
    
    # 添加自环
    A = A + torch.eye(num_nodes)
    
    return A


# ============================================================================
# 使用示例
# ============================================================================

'''if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    
    # ========== 1. 准备输入数据 ==========
    batch_size = 32
    num_nodes = 100
    feature_dim = 64
    
    # 历史记忆特征 (TCN编码后)
    X_h = torch.randn(batch_size, num_nodes, feature_dim)
    
    # 当前观测特征
    X_c = torch.randn(batch_size, num_nodes, feature_dim)
    
    # 邻接矩阵 (随机生成示例)
    A = torch.randn(num_nodes, num_nodes)
    A = (A + A.T) / 2          # 对称化
    A = (A > 0).float()        # 二值化
    A = A + torch.eye(num_nodes)  # 添加自环
    
    # ========== 2. 初始化STAFF模块 ==========
    staff = STAFFModule(
        in_dim=feature_dim,
        hidden_dim=64,
        gamma_min=0.1,
        gamma_max=0.5,
        gamma_init=0.3,
        curvature=1.0
    )
    
    # 打印模块信息
    print("=" * 60)
    print("STAFF v3.0 模块信息")
    print("=" * 60)
    print(f"输入特征维度: {feature_dim}")
    print(f"门控隐藏层维度: {64}")
    print(f"空间滤波系数γ: [{staff.gamma_min}, {staff.gamma_max}]")
    print(f"初始γ值: {staff.get_gamma():.4f}")
    print(f"双曲曲率c: {staff.get_curvature()}")
    print(f"可学习参数量: {sum(p.numel() for p in staff.parameters()):,}")
    print("=" * 60)
    
    # ========== 3. 前向传播 ==========
    X_fusion, alpha = staff(X_h, X_c, A)
    
    # ========== 4. 输出验证 ==========
    print("\n输出验证:")
    print(f"  融合特征形状: {X_fusion.shape}")
    print(f"  门控权重形状: {alpha.shape}")
    print(f"  门控权重范围: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"  门控权重均值: {alpha.mean():.4f}")
    
    # ========== 5. 训练示例 ==========
    print("\n" + "=" * 60)
    print("训练示例")
    print("=" * 60)
    
    # 优化器
    optimizer = torch.optim.Adam(staff.parameters(), lr=0.001)
    
    # 模拟训练步骤
    for step in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        X_out, alpha = staff(X_h, X_c, A)
        
        # 模拟损失 (实际使用时替换为真实损失)
        loss = X_out.mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.6f}, γ = {staff.get_gamma():.4f}")
    
    print("\n✅ STAFF 模块运行成功!")'''