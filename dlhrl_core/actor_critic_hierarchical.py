import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HierarchicalActorNetwork(nn.Module):
    """
    分层 Actor 网络
    对应 LaTeX Section V-A
    支持 High-Level 和 Low-Level 两种模式
    """
    
    def __init__(self, level: str = 'high',
             state_dim_high: int = 16,
             state_dim_low: int = 15,
             action_dim: int = 2,
             hidden_dim: int = 64):
        super(HierarchicalActorNetwork, self).__init__()
        self.level = level
        self.action_dim = action_dim
        
        print(f"[DEBUG] Actor {level} initialized: state_dim={state_dim_high if level=='high' else state_dim_low}, hidden_dim={hidden_dim}")
        
        if level == 'high':
            state_dim = state_dim_high
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Sigmoid()
            )
        else:
            state_dim = state_dim_low
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)
    
    def select_action(self, state: torch.Tensor,
                      explore: bool = True,
                      noise_scale: float = 0.1) -> torch.Tensor:
        """
        选择动作（带探索噪声）
        """
        action = self.forward(state)
        
        if explore:
            noise = torch.randn_like(action) * noise_scale
            action = action + noise
            action = torch.clamp(action, -1, 1)
        
        return action


class HierarchicalCriticNetwork(nn.Module):
    """
    分层 Critic 网络
    评估状态 - 动作对的价值
    """
    
    def __init__(self, level: str = 'high',
                 state_dim_high: int = 16,      # 【修复】与 Actor 一致
                 state_dim_low: int = 15,       # 【修复】与 Actor 一致
                 action_dim: int = 2,
                 hidden_dim: int = 64):         # 【修复】降低隐藏层维度
        super(HierarchicalCriticNetwork, self).__init__()
        self.level = level
        
        if level == 'high':
            state_dim = state_dim_high
        else:
            state_dim = state_dim_low
        
        # Q(s, a) 网络
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q 值
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TargetNetworkMixin:
    """目标网络软更新混合类"""
    
    def soft_update(self, source_network: nn.Module,
                    target_network: nn.Module,
                    tau: float = 0.005):
        """
        软更新目标网络
        θ' ← τθ + (1-τ)θ'
        """
        for target_param, param in zip(target_network.parameters(),
                                        source_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)