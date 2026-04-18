import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import cvxpy as cp

class LSOModule:
    """
    Lyapunov Safety Optimization (LSO) 模块
    对应 LaTeX Section IV-D
    功能：基于离散时间 CLF-CBF 的安全动作优化
    """
    
    def __init__(self, v_max: float = 2.0, theta_max: float = 0.5,
                 c_clf: float = 0.1, gamma_cbf: float = 0.5,
                 epsilon_init: float = 0.95, epsilon_final: float = 0.01,
                 epsilon_decay: float = 40.0):
        """
        参数:
            v_max: 最大速度
            theta_max: 最大转向角
            c_clf: CLF 参数 c (Eq. 20)
            gamma_cbf: CBF 参数γ (Eq. 21)
            epsilon_init: 初始 LSO 因子 (Eq. 26)
            epsilon_final: 最终 LSO 因子
            epsilon_decay: 衰减率
        """
        self.v_max = v_max
        self.theta_max = theta_max
        self.c_clf = c_clf
        self.gamma_cbf = gamma_cbf
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
    
    def compute_clf(self, position: torch.Tensor,
                    goal_position: torch.Tensor) -> torch.Tensor:
        """
        计算控制 Lyapunov 函数 V(x_t)
        对应 LaTeX Eq. (20)
        """
        V = torch.sum((position - goal_position) ** 2, dim=-1, keepdim=True)
        return V
    
    def compute_cbf(self, position: torch.Tensor,
                    obstacle_positions: torch.Tensor,
                    obstacle_radii: torch.Tensor = None) -> torch.Tensor:
        """
        计算控制屏障函数 B(x_t)
        对应 LaTeX Eq. (21)
        """
        B = float('inf')
        for i in range(obstacle_positions.shape[1]):
            dist_sq = torch.sum((position - obstacle_positions[:, i:i+1, :]) ** 2, dim=-1, keepdim=True)
            r_sq = (obstacle_radii[i] ** 2) if obstacle_radii is not None else 1.0
            B_i = dist_sq - r_sq
            B = torch.minimum(B if B != float('inf') else B_i, B_i)
        return B
    
    def solve_qcqp(self, a_nom: torch.Tensor,
               position: torch.Tensor,
               goal_position: torch.Tensor,
               obstacle_positions: torch.Tensor,
               traversability: torch.Tensor = None) -> Tuple[torch.Tensor, bool]:
        """
        求解 QCQP 优化问题获取安全动作
        对应 LaTeX Eq. (25)
        
        返回:
          a_safe: 安全动作 [v, θ]
          is_safe: 是否安全标志
        """
        B = a_nom.shape[0]
        device = a_nom.device  # 【修复】获取正确设备
        a_safe = torch.zeros_like(a_nom)
        is_safe = torch.ones(B, 1, dtype=torch.bool, device=device)  # 【修复】指定设备
        
        # 【修复】确保所有输入张量在同一设备上
        position = position.to(device)
        goal_position = goal_position.to(device)
        obstacle_positions = obstacle_positions.to(device)
        
        # 简化实现：基于规则的安全检查
        for b in range(B):
          v_nom = a_nom[b, 0].item()
          theta_nom = a_nom[b, 1].item()
          
          # 检查障碍物距离
          min_dist = float('inf')
          for i in range(obstacle_positions.shape[1]):
              # 【修复】确保在同一设备上计算
              dist = torch.norm(position[b].to(device) - obstacle_positions[b, i].to(device)).item()
              min_dist = min(min_dist, dist)
          
          # 如果距离过近，调整动作
          if min_dist < 5.0:
              # 减速并转向
              v_safe = v_nom * 0.5
              theta_safe = theta_nom + 0.3 * np.sign(np.random.randn())
              is_safe[b, 0] = False
          else:
              v_safe = v_nom
              theta_safe = theta_nom
          
          # 应用约束
          v_safe = np.clip(v_safe, 0, self.v_max)
          theta_safe = np.clip(theta_safe, -self.theta_max, self.theta_max)
          
          a_safe[b, 0] = v_safe
          a_safe[b, 1] = theta_safe
        
        return a_safe, is_safe
    
    def get_safe_action(self, a_nom: torch.Tensor,
                    lso_input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取安全动作
        对应 LaTeX Alg. 2
        
        返回:
            a_safe: 安全动作
            is_safe_action: 安全标志
        """
        device = a_nom.device  # 【修复】获取正确设备
        
        position = lso_input.get('position', torch.zeros(a_nom.shape[0], 2, device=device))
        position = position.to(device)  # 【修复】确保设备一致
        
        goal_position = lso_input.get('goal_position', torch.ones(a_nom.shape[0], 2, device=device) * 50)
        goal_position = goal_position.to(device)  # 【修复】确保设备一致
        
        obstacle_positions = lso_input.get('obstacle_positions',
                                            torch.zeros(a_nom.shape[0], 10, 2, device=device))
        obstacle_positions = obstacle_positions.to(device)  # 【修复】确保设备一致
        
        traversability = lso_input.get('traversability', None)
        
        a_safe, is_safe = self.solve_qcqp(
            a_nom, position, goal_position, obstacle_positions, traversability
        )
        
        return a_safe, is_safe
    
    def get_epsilon_lso(self, episode: int) -> float:
        """
        获取自适应 LSO 因子
        对应 LaTeX Eq. (26)
        """
        epsilon = self.epsilon_final + (self.epsilon_init - self.epsilon_final) / \
                  (1 + np.exp(episode / self.epsilon_decay))
        return epsilon