import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class DRMGModule:
    """
    Drift-Regularized Meta-Gradient (DRMG) 模块
    对应 LaTeX Section V-B Eq. (27-29) 和 Alg. 3
    功能：漂移正则化元梯度奖励自适应
    """
    
    def __init__(self, num_reward_components: int = 4,
                 lambda_reg: float = 0.1,
                 phi_min: float = 0.05,
                 phi_max: float = 0.95,
                 beta: float = 0.001,
                 device: str = 'cpu'):
        """
        参数:
            num_reward_components: 奖励分量数量 (默认 4)
            lambda_reg: 漂移正则化系数 (Eq. 28)
            phi_min, phi_max: φ参数范围
            beta: 元梯度学习率
            device: 设备
        """
        self.num_reward_components = num_reward_components
        self.lambda_reg = lambda_reg
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.beta = beta
        self.device = device
        
        # 初始化奖励参数φ
        self.phi = torch.ones(num_reward_components, device=device) / num_reward_components
    
    def compute_reward(self, state: Dict[str, torch.Tensor],
                       action: torch.Tensor,
                       is_safe_action: torch.Tensor,
                       battery: torch.Tensor,
                       E_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算参数化奖励 R_s(φ)
        对应 LaTeX Eq. (27)
        
        返回:
            R_s: 密集奖励
            r_components: 各奖励分量 [r1, r2, r3, r4]
        """
        B = action.shape[0]
        
        # r1 = -E(t) 能量消耗
        r1 = -state.get('energy_consumption', torch.ones(B, 1) * 0.1)
        
        # r2 = P_LSO - E(t) LSO 干预
        P_LSO = -5.0
        r2 = P_LSO * (1 - is_safe_action.float()) - state.get('energy_consumption', torch.ones(B, 1) * 0.1)
        
        # r3 = P_goal - E(t) 任务完成
        P_goal = 10.0
        task_completed = state.get('task_completed', torch.zeros(B, 1))
        r3 = P_goal * task_completed - state.get('energy_consumption', torch.ones(B, 1) * 0.1)
        
        # r4 = max(E_bound - be_n, 0) 电量安全
        r4 = torch.clamp(E_bound - battery, min=0)
        
        r_components = torch.cat([r1, r2, r3, r4], dim=-1)  # [B, 4]
        
        # R_s(φ) = Σφ_i * r_i
        R_s = torch.sum(self.phi.unsqueeze(0).to(r_components.device) * r_components, dim=-1, keepdim=True)
        
        return R_s, r_components
    
    def compute_advantage(self, rewards: torch.Tensor,
                          values: torch.Tensor,
                          gamma: float = 0.95) -> torch.Tensor:
        """
        计算优势函数 A^π
        对应 LaTeX Eq. (29)
        """
        advantages = rewards - values
        return advantages
    
    def compute_meta_gradient(self, transitions: List[Dict],
                               lyapunov_drift: torch.Tensor,
                               phi: torch.Tensor = None) -> torch.Tensor:
        """
        计算元梯度 ∇_φ J
        对应 LaTeX Eq. (28) 和 Alg. 3 Line 21-32
        
        参数:
            transitions: 轨迹数据列表
            lyapunov_drift: Δ(Θ(t)) 来自 LAO
            phi: 当前奖励参数
            
        返回:
            meta_gradient: 元梯度
        """
        if phi is None:
            phi = self.phi
        
        # Term 1: 仅使用非干预样本 (Eq. 29)
        filtered_transitions = [t for t in transitions if not t.get('is_safe_action', False)]
        
        if len(filtered_transitions) > 0:
            r_components_filtered = torch.stack([t['r_components'] for t in filtered_transitions])
            advantages_filtered = torch.stack([t['advantage'] for t in filtered_transitions])
            grad_term1 = torch.mean(r_components_filtered * advantages_filtered, dim=0)
        else:
            grad_term1 = torch.zeros(self.num_reward_components, device=lyapunov_drift.device)
        
        # Term 2: 使用全部样本 (漂移正则化)
        r_components_all = torch.stack([t['r_components'] for t in transitions])
        grad_term2 = torch.mean(r_components_all, dim=0)
        
        # 最终元梯度 (Eq. 28)
        meta_gradient = grad_term1 - self.lambda_reg * lyapunov_drift * grad_term2
        
        return meta_gradient
    
    def update_reward_params(self, transitions: List[Dict],
                              lyapunov_drift: torch.Tensor,
                              phi: torch.Tensor = None) -> torch.Tensor:
        """
        更新奖励参数φ
        对应 LaTeX Alg. 3 Line 21-32
        
        返回:
            phi_updated: 更新后的奖励参数
        """
        if phi is None:
            phi = self.phi.clone()
        
        # 计算元梯度
        meta_gradient = self.compute_meta_gradient(transitions, lyapunov_drift, phi)
        
        # 更新φ
        phi_updated = phi + self.beta * meta_gradient
        
        # 约束φ范围
        phi_updated = torch.clamp(phi_updated, self.phi_min, self.phi_max)
        
        # 归一化 (L1 归一化)
        phi_updated = phi_updated / phi_updated.sum()
        
        return phi_updated
    
    def get_phi(self) -> torch.Tensor:
        """获取当前奖励参数"""
        return self.phi.clone()
    
    def set_phi(self, phi: torch.Tensor):
        """设置奖励参数"""
        self.phi = phi.clone()