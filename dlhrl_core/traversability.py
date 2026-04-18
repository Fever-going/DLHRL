import torch
import torch.nn as nn
import numpy as np

class TraversabilityCalculator:
    """
    可通行性计算器
    对应 LaTeX Eq. (14): P_x,y = w1*P_map + w2*Σ(R_j,t * P_cam)
    """
    
    def __init__(self, map_size: int = 50, w1: float = 0.6, w2: float = 0.4):
        self.map_size = map_size
        self.w1 = w1
        self.w2 = w2
    
    def calculate(self, P_map: torch.Tensor,
                  R_jt: torch.Tensor,
                  P_cam: torch.Tensor,
                  obstacle_positions: torch.Tensor = None) -> torch.Tensor:
        """
        计算实时可通行性
        
        参数:
            P_map: 全局地形数据 [H, W]
            R_jt: 碰撞风险 [B, N_risk, 1]
            P_cam: 局部地形数据 [B, H_cam, W_cam]
            obstacle_positions: 障碍物位置 [B, N_risk, 2]
            
        返回:
            P_xy: 可通行性 [B, H, W]
        """
        B = R_jt.shape[0]
        H, W = P_map.shape
        
        # 基础地形
        P_xy = self.w1 * P_map.unsqueeze(0).expand(B, -1, -1)
        
        # 融合风险数据
        if obstacle_positions is not None:
            # 将障碍物风险投影到地图
            risk_map = self._project_risk_to_map(
                obstacle_positions, R_jt, H, W
            )
            P_xy = P_xy + self.w2 * risk_map
        
        P_xy = torch.clamp(P_xy, 0.0, 1.0)
        return P_xy
    
    def _project_risk_to_map(self, positions: torch.Tensor,
                              risks: torch.Tensor,
                              H: int, W: int) -> torch.Tensor:
        """将风险投影到地图网格"""
        B, N, _ = positions.shape
        risk_map = torch.zeros(B, H, W, device=positions.device)
        
        for b in range(B):
            for n in range(N):
                x = int(positions[b, n, 0].item() / W * H)
                y = int(positions[b, n, 1].item() / W * W)
                x = min(max(x, 0), H - 1)
                y = min(max(y, 0), W - 1)
                risk_map[b, x, y] = risks[b, n, 0]
        
        return risk_map