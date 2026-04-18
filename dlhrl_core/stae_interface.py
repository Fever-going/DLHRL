import torch
import numpy as np
from typing import Dict, Tuple, Optional

class STAEDataInterface:
    """
    STAE 数据接口层
    功能：将 DTGAT 输出转换为 LAO 和 LSO 模块的输入数据
    对应 LaTeX Section IV-B STAE Module
    """
    
    def __init__(self, num_task_types: int = 5, num_risk_nodes: int = 20,
                 w1: float = 0.6, w2: float = 0.4):
        """
        参数:
            num_task_types: 任务类型数量 S
            num_risk_nodes: 风险节点数量
            w1, w2: 可通行性计算权重 (Eq. 14)
        """
        self.num_task_types = num_task_types
        self.num_risk_nodes = num_risk_nodes
        self.w1 = w1
        self.w2 = w2
    
    def process_task_importance(self, I_it_hat: torch.Tensor,
                                 queue_state: Dict[str, torch.Tensor],
                                 task_arrivals: torch.Tensor,
                                 energy_estimate: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        处理任务重要性，生成 LAO 输入数据
        对应 LaTeX Eq. (13-15)
        
        参数:
            I_it_hat: DTGAT 输出的任务重要性 [B, N_task, 1]
            queue_state: 队列状态 {R_i, B_i, ϑ}
            task_arrivals: 任务到达 D_i(t)
            energy_estimate: 能量估计 E_i,j
            
        返回:
            task_data: LAO 输入数据字典
        """
        B, N_task, _ = I_it_hat.shape
        
        task_data = {
            'importance': I_it_hat,  # Î_i,t
            'queue_R': queue_state.get('R', torch.zeros(B, self.num_task_types, 1)),
            'queue_B': queue_state.get('B', torch.zeros(B, self.num_task_types, 1)),
            'queue_theta': queue_state.get('theta', torch.zeros(B, 1, 1)),
            'task_arrivals': task_arrivals,  # D_i(t)
            'energy_estimate': energy_estimate  # E_i,j
        }
        
        return task_data
    
    def compute_traversability(self, R_jt: torch.Tensor,
                                P_map: torch.Tensor,
                                P_cam: torch.Tensor) -> torch.Tensor:
        """
        计算可通行性 P_x,y
        对应 LaTeX Eq. (14)
        
        参数:
            R_jt: DTGAT 输出的碰撞风险 [B, N_risk, 1]
            P_map: 全局地形数据 [H, W]
            P_cam: 局部观测地形数据 [B, H_cam, W_cam]
            
        返回:
            P_xy: 可通行性数据 [B, H, W]
        """
        B = R_jt.shape[0]
        H, W = P_map.shape
        
        # 初始化可通行性
        P_xy = self.w1 * P_map.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        
        # 融合碰撞风险与局部地形
        if P_cam is not None and R_jt is not None:
            # 简化：将风险映射到地形网格
            risk_weight = R_jt.mean(dim=1, keepdim=True)  # [B, 1, 1]
            P_cam_expanded = torch.nn.functional.interpolate(
                P_cam.unsqueeze(1),  # [B, 1, H_cam, W_cam]
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H, W]
            
            P_xy = P_xy + self.w2 * risk_weight * P_cam_expanded
        
        # 归一化到 [0, 1]
        P_xy = torch.clamp(P_xy, 0.0, 1.0)
        
        return P_xy
    
    def get_lao_input(self, I_it_hat: torch.Tensor,
                      env_state: Dict) -> Dict[str, torch.Tensor]:
        """
        生成 LAO 模块的完整输入数据
        
        参数:
            I_it_hat: DTGAT 输出的任务重要性
            env_state: 环境状态字典
            
        返回:
            lao_input: LAO 输入数据
        """
        queue_state = {
            'R': env_state.get('queue_R', None),
            'B': env_state.get('queue_B', None),
            'theta': env_state.get('queue_theta', None)
        }
        
        task_arrivals = env_state.get('task_arrivals',
                                       torch.zeros(I_it_hat.shape[0], self.num_task_types, 1))
        energy_estimate = env_state.get('energy_estimate',
                                         torch.ones(I_it_hat.shape[0], 1, 1) * 0.5)
        
        return self.process_task_importance(I_it_hat, queue_state,
                                            task_arrivals, energy_estimate)
    
    def get_lso_input(self, R_jt: torch.Tensor,
                      env_state: Dict) -> Dict[str, torch.Tensor]:
        """
        生成 LSO 模块的完整输入数据
        
        参数:
            R_jt: DTGAT 输出的碰撞风险
            env_state: 环境状态字典
            
        返回:
            lso_input: LSO 输入数据
        """
        P_map = env_state.get('P_map', torch.ones(50, 50) * 0.8)
        P_cam = env_state.get('P_cam', None)
        
        P_xy = self.compute_traversability(R_jt, P_map, P_cam)
        
        lso_input = {
            'collision_risk': R_jt,
            'map_terrain': P_map,
            'camera_terrain': P_cam,
            'traversability': P_xy,
            'obstacle_positions': env_state.get('obstacle_positions', None)
        }
        
        return lso_input