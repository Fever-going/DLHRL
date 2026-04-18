import torch
import numpy as np

class RobotDynamics:
    """
    机器人运动学模型
    对应 LaTeX Eq. (1)
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
    
    def update_position(self, position: torch.Tensor,
                        velocity: torch.Tensor,
                        angle: torch.Tensor) -> torch.Tensor:
        """
        更新位置
        x(t+1) = x(t) + v*Δt*sin(θ)
        y(t+1) = y(t) + v*Δt*cos(θ)
        """
        x_new = position[..., 0] + velocity * self.dt * torch.sin(angle)
        y_new = position[..., 1] + velocity * self.dt * torch.cos(angle)
        return torch.stack([x_new, y_new], dim=-1)
    
    def update_battery(self, battery: torch.Tensor,
                       velocity: torch.Tensor,
                       dt: float = None) -> torch.Tensor:
        """更新电量"""
        if dt is None:
            dt = self.dt
        consumption = 0.1 * velocity + 0.05
        return battery - consumption * dt