import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class LAOModule:
    """
    Lyapunov Allocation Optimization (LAO) 模块
    对应 LaTeX Section IV-C
    功能：基于 Lyapunov 优化的任务分配与队列稳定性管理
    """
    
    def __init__(self, num_task_types: int = 5, V_param: float = 1.0,
                 lambda_V: float = 0.5, I_QRM: float = 10.0):
        """
        参数:
            num_task_types: 任务类型数量 S
            V_param: Lyapunov 优化参数 V (Eq. 18)
            lambda_V: 紧急性权重 (Eq. 18)
            I_QRM: 最大任务重要性积压长度
        """
        self.num_task_types = num_task_types
        self.V_param = V_param
        self.lambda_V = lambda_V
        self.I_QRM = I_QRM
    
    def initialize_queues(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """初始化任务队列"""
        return {
            'R': torch.zeros(batch_size, self.num_task_types, 1, device=device),
            'B': torch.zeros(batch_size, self.num_task_types, 1, device=device),
            'theta': torch.zeros(batch_size, 1, 1, device=device)
        }
    
    def update_queues(self, queue_state: Dict[str, torch.Tensor],
                      task_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        更新任务队列
        对应 LaTeX Eq. (13-15)
        """
        R = queue_state['R']
        B = queue_state['B']
        theta = queue_state['theta']
        
        D_i = task_data['task_arrivals']  # D_i(t)
        I_it = task_data['importance']    # Î_i,t
        O_i = task_data.get('O_i', torch.zeros_like(D_i))  # O_i(t)
        c_i = task_data.get('c_i', torch.zeros_like(D_i))  # c_i(t)
        
        # Eq. (13): R_i(t+1)
        R_next = torch.clamp(R - O_i, min=0) + D_i
        
        # Eq. (14): B_i(t+1)
        B_next = torch.clamp(B - c_i, min=0) + O_i
        
        # Eq. (15): ϑ(t+1)
        h_i = I_it.mean(dim=1, keepdim=True)  # h(i)
        theta_next = torch.clamp(theta - h_i, min=0) + I_it.mean(dim=1, keepdim=True)
        
        return {
            'R': R_next,
            'B': B_next,
            'theta': theta_next
        }
    
    def compute_lyapunov_function(self, queue_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 Lyapunov 函数 L(Θ(t))
        对应 LaTeX Eq. (16)
        """
        R = queue_state['R']
        B = queue_state['B']
        theta = queue_state['theta']
        
        L = 0.5 * (torch.sum(R ** 2, dim=[1, 2]) +
                   torch.sum(B ** 2, dim=[1, 2]) +
                   theta.squeeze(-1).squeeze(-1) ** 2)
        return L
    
    def compute_lyapunov_drift(self, queue_state: Dict[str, torch.Tensor],
                                queue_next: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 Lyapunov 漂移 Δ(Θ(t))
        对应 LaTeX Eq. (17)
        """
        L_t = self.compute_lyapunov_function(queue_state)
        L_t1 = self.compute_lyapunov_function(queue_next)
        drift = L_t1 - L_t
        return drift
    
    def optimize_task_allocation(self, task_data: Dict[str, torch.Tensor],
                                  queue_state: Dict[str, torch.Tensor],
                                  energy_constraint: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        优化任务分配决策
        对应 LaTeX Eq. (18-19) 和 Alg. 1
        
        返回:
            c_i: 任务执行决策
            allocation_info: 分配信息字典
        """
        R = queue_state['R']
        B = queue_state['B']
        theta = queue_state['theta']
        I_it = task_data['importance']
        
        # Drift-plus-penalty 优化
        # 简化实现：基于队列状态和任务重要性选择任务
        priority = B * I_it + R * 0.5 + theta * self.lambda_V
        
        # 能量约束检查
        energy_available = energy_constraint > 0.5
        c_i = (priority > priority.median(dim=1, keepdim=True)[0]).float() * energy_available.float()
        
        allocation_info = {
            'priority': priority,
            'energy_sufficient': energy_available,
            'num_tasks_selected': c_i.sum(dim=1)
        }
        
        return c_i, allocation_info
    
    def check_energy_constraint(self, battery_level: torch.Tensor,
                                 E_bound: torch.Tensor) -> torch.Tensor:
        """
        检查能量约束
        对应 LaTeX Eq. (12)
        """
        return battery_level >= E_bound