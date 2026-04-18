import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

# 添加 ablation_staff 路径以复用 DTGAT 模型
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ablation_staff'))
from model_unified import DTGATModel

from .stae_interface import STAEDataInterface
from .lao_module import LAOModule
from .lso_module import LSOModule
from .drmg_module import DRMGModule
from .actor_critic_hierarchical import HierarchicalActorNetwork, HierarchicalCriticNetwork, TargetNetworkMixin
from .replay_buffer import DualReplayBuffer

class DLHRLAgent(TargetNetworkMixin):
    """
    DLHRL 智能体主类
    整合 LAO + LSO + DRMG + 分层 Actor-Critic + 双缓冲区
    对应 LaTeX Alg. 3
    """
    
    def __init__(self,
                 # DTGAT 配置
                 dtgat_model_path: str = None,
                 freeze_dtgat: bool = True,
                 
                 # 网络配置
                 state_dim_high: int = 128,
                 state_dim_low: int = 64,
                 action_dim: int = 2,
                 hidden_dim: int = 128,
                 
                 # RL 配置
                 lr: float = 0.001,
                 gamma: float = 0.95,
                 tau: float = 0.005,
                 
                 # 双缓冲区配置
                 lambda_RF: float = 1.0,
                 capacity_demo: int = 5000,
                 capacity_interaction: int = 20000,
                 
                 # DRMG 配置
                 lambda_reg: float = 0.1,
                 chi: int = 40,  # 元梯度更新频率
                 
                 # LSO 配置
                 epsilon_lso_init: float = 0.95,
                 epsilon_lso_final: float = 0.01,
                 
                 # 设备
                 device: str = 'cuda'):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.chi = chi
        
        # 1. 加载 DTGAT 模型 (STAE 模块)
        # 1. 加载 DTGAT 模型 (STAE 模块)
        self.dtgat = DTGATModel(use_staff=True)
        if dtgat_model_path and os.path.exists(dtgat_model_path):
            self.dtgat.load_state_dict(torch.load(dtgat_model_path, map_location=self.device))
            print(f"✓ Loaded DTGAT from {dtgat_model_path}")
            
            # 【修复】确保属性存在（兼容旧版本模型）
            if not hasattr(self.dtgat, 'num_nodes_task'):
                self.dtgat.num_nodes_task = 5  # 默认值
            if not hasattr(self.dtgat, 'num_nodes_risk'):
                self.dtgat.num_nodes_risk = 20  # 默认值
        else:
            print("⚠ DTGAT model path not found, using random initialization")
        
        if freeze_dtgat:
            for param in self.dtgat.parameters():
                param.requires_grad = False
            self.dtgat.eval()
        
        self.dtgat.to(self.device)
        
        # 2. STAE 接口
        self.stae_interface = STAEDataInterface(
            num_task_types=5,      # 必须与 DTGAT 的 num_nodes_task 一致
            num_risk_nodes=20      # 与 DTGAT 的 num_nodes_risk 一致
        )
        # 3. LAO 模块
        self.lao = LAOModule()
        self.queue_state = None
        
        # 4. LSO 模块
        self.lso = LSOModule(epsilon_init=epsilon_lso_init,
                             epsilon_final=epsilon_lso_final)
        
        # 5. DRMG 模块
        self.drmg = DRMGModule(lambda_reg=lambda_reg, device=device)
        
        # 6. 分层 Actor-Critic 网络
        self.actor_high = HierarchicalActorNetwork(level='high',
                                                    state_dim_high=16,      # queue_R(5)+queue_B(5)+importance(5)+power(1)
                                                    action_dim=action_dim,
                                                    hidden_dim=64).to(self.device)
        self.actor_low = HierarchicalActorNetwork(level='low',
                                                   state_dim_low=2,  # position only
                                                   action_dim=action_dim,
                                                   hidden_dim=64).to(self.device)
        self.critic_high = HierarchicalCriticNetwork(level='high',
                                                      state_dim_high=16,
                                                      action_dim=action_dim,
                                                      hidden_dim=64).to(self.device)
        self.critic_low = HierarchicalCriticNetwork(level='low',
                                                      state_dim_low=2,  # position only for now
                                                     hidden_dim=64).to(self.device)
        
        # 目标网络
        self.actor_high_target = HierarchicalActorNetwork(level='high',
                                                           state_dim_high=16,
                                                           action_dim=action_dim,
                                                           hidden_dim=64).to(self.device)
        self.actor_low_target = HierarchicalActorNetwork(level='low',
                                                          state_dim_low=2,
                                                          action_dim=action_dim,
                                                          hidden_dim=64).to(self.device)
        self.critic_high_target = HierarchicalCriticNetwork(level='high',
                                                             state_dim_high=16,
                                                             action_dim=action_dim,
                                                             hidden_dim=64).to(self.device)
        self.critic_low_target = HierarchicalCriticNetwork(level='low',
                                                            state_dim_low=2,
                                                            action_dim=action_dim,
                                                            hidden_dim=64).to(self.device)
        
        # 初始化目标网络
        self.actor_high_target.load_state_dict(self.actor_high.state_dict())
        self.actor_low_target.load_state_dict(self.actor_low.state_dict())
        self.critic_high_target.load_state_dict(self.critic_high.state_dict())
        self.critic_low_target.load_state_dict(self.critic_low.state_dict())
        
        # 7. 优化器
        self.actor_optimizer = optim.Adam(
            list(self.actor_high.parameters()) + list(self.actor_low.parameters()),
            lr=lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_high.parameters()) + list(self.critic_low.parameters()),
            lr=lr
        )
        
        # 8. 双缓冲区
        self.replay_buffer = DualReplayBuffer(
            capacity_demo=capacity_demo,
            capacity_interaction=capacity_interaction,
            lambda_RF=lambda_RF
        )
        
        # 9. 训练统计
        self.step_count = 0
        self.episode_count = 0
        self.training_logs = []
    
    def get_stae_features(self, env_observation: Dict) -> Tuple[Dict, Dict]:
        """
        从环境观测获取 STAE 特征
        对应 LaTeX Alg. 3 Line 4-5
        """
        # 提取 DTGAT 输入
        obstacles = env_observation['obstacles'].to(self.device)
        history = env_observation['robot_history'].to(self.device)
        current = env_observation['robot_current'].to(self.device)
        adj_task = env_observation['task_adjacency'].to(self.device)
        node_mask = env_observation.get('node_mask', None)
        if node_mask is not None:
            node_mask = node_mask.to(self.device)
        
        # DTGAT 前向传播
        with torch.no_grad():
            I_it, R_jt, alpha = self.dtgat(
                obstacles, history, current, adj_task, node_mask
            )
        
        # 【修复】构建完整的 env_state，包含队列状态
        env_state = env_observation.copy()
        if self.queue_state is not None:
            env_state['queue_R'] = self.queue_state['R']
            env_state['queue_B'] = self.queue_state['B']
            env_state['queue_theta'] = self.queue_state['theta']
        else:
            # 初始化队列状态
            self.queue_state = self.lao.initialize_queues(
                env_observation['obstacles'].shape[0], 
                self.device
            )
            env_state['queue_R'] = self.queue_state['R']
            env_state['queue_B'] = self.queue_state['B']
            env_state['queue_theta'] = self.queue_state['theta']
        
        # 通过 STAE 接口转换
        task_data = self.stae_interface.get_lao_input(I_it, env_state)
        trav_data = self.stae_interface.get_lso_input(R_jt, env_state)
        
        return task_data, trav_data
    
    def select_action(self, state: Dict[str, torch.Tensor],
                  task_data: Dict, trav_data: Dict,
                  episode: int, explore: bool = True) -> Tuple[torch.Tensor, bool]:
        """
        选择动作（含 LSO 检查）
        对应 LaTeX Alg. 2 和 Alg. 3 Line 6-10
        """
        # 【修复】确保 state 中的张量在正确设备上
        for key in state:
            if isinstance(state[key], torch.Tensor) and state[key].device != self.device:
                state[key] = state[key].to(self.device)
        
        # 1. High-Level 决策（任务分配）
        high_state = self._encode_high_state(state, task_data)
        high_action = self.actor_high.select_action(high_state, explore=explore)
        
        # 2. Low-Level 决策（运动控制）
        low_state = self._encode_low_state(state, trav_data)
        low_action_nom = self.actor_low.select_action(low_state, explore=explore)
        
        # 3. LSO 安全检查
        epsilon_lso = self.lso.get_epsilon_lso(episode)
        if np.random.rand() < epsilon_lso:
            # 使用 LSO 安全动作
            trav_data['position'] = state['position']
            trav_data['goal_position'] = state.get('goal_position', torch.ones_like(state['position']) * 50)
            
            # 【修复】确保 goal_position 在正确设备上
            if trav_data['goal_position'].device != self.device:
                trav_data['goal_position'] = trav_data['goal_position'].to(self.device)
            
            low_action_safe, is_safe = self.lso.get_safe_action(low_action_nom, trav_data)
        else:
            # 使用 Actor 输出
            low_action_safe = low_action_nom
            is_safe = torch.zeros_like(low_action_nom[:, :1], dtype=torch.bool)
        
        # 合并动作
        action = torch.cat([high_action, low_action_safe], dim=-1)
        
        return action, is_safe
    
    def _encode_high_state(self, state: Dict, task_data: Dict) -> torch.Tensor:
        """编码 High-Level 状态"""
        # S_Task + S_Power + Î_i,t
        if task_data.get('queue_R') is None:
            batch_size = state['position'].shape[0]
            task_data['queue_R'] = torch.zeros(batch_size, 5, 1, device=self.device)
            task_data['queue_B'] = torch.zeros(batch_size, 5, 1, device=self.device)
            task_data['importance'] = torch.zeros(batch_size, 5, 1, device=self.device)
        
        # 【修复】确保 importance 维度与 queue 一致 (DTGAT 输出 20 节点，需要聚合到 5 任务类型)
        importance = task_data['importance']  # [B, 20, 1] 或 [B, 5, 1]
        if importance.shape[1] != 5:
            # 将 20 个节点聚合到 5 个任务类型 (平均池化)
            importance = importance.view(importance.shape[0], 5, 4, -1).mean(dim=2)  # [B, 5, 1]
        
        queue_R = task_data['queue_R'].view(state['position'].shape[0], -1)  # [B, 5]
        queue_B = task_data['queue_B'].view(state['position'].shape[0], -1)  # [B, 5]
        importance = importance.view(state['position'].shape[0], -1)  # [B, 5]
        
        # 【修复】确保 power 张量在正确设备上
        power = state.get('power', torch.ones(state['position'].shape[0], 1, device=self.device))
        if power.device != self.device:
            power = power.to(self.device)
        
        high_state = torch.cat([queue_R, queue_B, importance, power], dim=-1)  # [B, 16]
        
        # 验证维度
        expected_dim = 16
        actual_dim = high_state.shape[-1]
        if actual_dim != expected_dim:
            print(f"⚠ Warning: High-level state dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            if actual_dim > expected_dim:
                high_state = high_state[:, :expected_dim]
            else:
                padding = torch.zeros(state['position'].shape[0], expected_dim - actual_dim, device=self.device)
                high_state = torch.cat([high_state, padding], dim=-1)
        
        return high_state
    
    def _encode_low_state(self, state: Dict, trav_data: Dict) -> torch.Tensor:
        """编码 Low-Level 状态"""
        # S_rp + Traversability
        position = state['position']
        
        # 【修复】确保 position 在正确设备上
        if position.device != self.device:
            position = position.to(self.device)
        
        # 暂时只使用 position
        low_state = position  # [B, 2]
        return low_state
    
    def store_transition(self, transition: Dict, is_safe_action: torch.Tensor):
        """存储交互数据"""
        for b in range(transition['state']['position'].shape[0]):
            single_transition = {
                'state': {k: v[b:b+1] for k, v in transition['state'].items()},
                'action': transition['action'][b:b+1],
                'reward': transition['reward'][b:b+1],
                'next_state': {k: v[b:b+1] for k, v in transition['next_state'].items()},
                'is_safe_action': is_safe_action[b].item(),
                'r_components': torch.zeros(4, device=self.device),
                'advantage': torch.zeros(1, device=self.device)
            }
            self.replay_buffer.store_transition(single_transition, is_safe_action[b].item())
    
    def update_networks(self, batch_size: int = 256):
        """
        更新 Actor/Critic 网络
        对应 LaTeX Alg. 3 Line 20
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # 采样批次
        batch = self.replay_buffer.sample_refined_batch(batch_size)
        
        # 转换为张量
        states = torch.cat([t['state']['position'].detach() for t in batch], dim=0).to(self.device)
        actions = torch.cat([t['action'].detach() for t in batch], dim=0).to(self.device)
        actions = actions[:, 2:]  # low action [v, θ]
        rewards = torch.cat([t['reward'].detach() for t in batch], dim=0).to(self.device)
        next_states = torch.cat([t['next_state']['position'].detach() for t in batch], dim=0).to(self.device)
        
        # Critic 更新
        current_q = self.critic_low(states, actions)
        with torch.no_grad():
            next_action = self.actor_low_target(next_states)
            target_q = self.critic_low_target(next_states, next_action)
            target_q = rewards + self.gamma * target_q
        
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 更新 - 暂时跳过，因为缺少完整状态
        # actor_action = self.actor_low(states)
        # actor_q = self.critic_low(states, actor_action)
        # actor_loss = -actor_q.mean()
        
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_low, self.actor_low_target, self.tau)
        self.soft_update(self.critic_low, self.critic_low_target, self.tau)
        
        return critic_loss.item(), 0.0  # actor_loss.item()
    
    def update_drmg_params(self, lyapunov_drift: torch.Tensor):
        """
        更新 DRMG 奖励参数φ
        对应 LaTeX Alg. 3 Line 21-32
        """
        if self.step_count % self.chi != 0 or self.step_count < 50:
            return
        
        # 采样批次用于元梯度
        batch = self.replay_buffer.sample_interaction_batch(256)
        
        # 准备转移数据
        transitions = []
        for t in batch:
            transitions.append({
                'r_components': t.get('r_components', torch.zeros(4)),
                'advantage': t.get('advantage', torch.zeros(1)),
                'is_safe_action': t.get('is_safe_action', False)
            })
        
        # 更新φ
        phi_updated = self.drmg.update_reward_params(transitions, lyapunov_drift)
        self.drmg.set_phi(phi_updated)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_high': self.actor_high.state_dict(),
            'actor_low': self.actor_low.state_dict(),
            'critic_high': self.critic_high.state_dict(),
            'critic_low': self.critic_low.state_dict(),
            'phi': self.drmg.get_phi(),
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_high.load_state_dict(checkpoint['actor_high'])
        self.actor_low.load_state_dict(checkpoint['actor_low'])
        self.critic_high.load_state_dict(checkpoint['critic_high'])
        self.critic_low.load_state_dict(checkpoint['critic_low'])
        self.drmg.set_phi(checkpoint['phi'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        print(f"✓ Model loaded from {path}")