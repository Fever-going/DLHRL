import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import random

class DualReplayBuffer:
    """
    双缓冲区架构
    对应 LaTeX Section V-B Fig. 1
    - B_Q^D (Safety Demo): LSO 生成的安全动作
    - B_Q^R (Interaction): 智能体 - 环境交互数据
    """
    
    def __init__(self, capacity_demo: int = 5000,
                 capacity_interaction: int = 20000,
                 lambda_RF: float = 1.0):
        """
        参数:
            capacity_demo: 安全演示缓冲区容量
            capacity_interaction: 交互缓冲区容量
            lambda_RF: 精炼因子 |B_Q^D| / |B_Q^R|
        """
        self.capacity_demo = capacity_demo
        self.capacity_interaction = capacity_interaction
        self.lambda_RF = lambda_RF
        
        self.demo_buffer = deque(maxlen=capacity_demo)
        self.interaction_buffer = deque(maxlen=capacity_interaction)
    
    def store_transition(self, transition: Dict,
                         is_safe_action: bool = False):
        """
        存储转移数据
        
        参数:
            transition: 转移数据字典
            is_safe_action: 是否为 LSO 安全动作
        """
        if is_safe_action:
            # 存储到安全演示缓冲区
            if len(self.demo_buffer) >= self.capacity_demo:
                self.demo_buffer.popleft()
            self.demo_buffer.append(transition)
        else:
            # 存储到交互缓冲区
            if len(self.interaction_buffer) >= self.capacity_interaction:
                self.interaction_buffer.popleft()
            self.interaction_buffer.append(transition)
    
    def sample_refined_batch(self, batch_size: int) -> List[Dict]:
        """
        采样精炼批次 B_Q = B_Q^D ∪ B_Q^R
        """
        demo_size = int(batch_size * self.lambda_RF / (1 + self.lambda_RF))
        interaction_size = batch_size - demo_size
        
        batch = []
        
        # 从演示缓冲区采样
        if len(self.demo_buffer) > 0:
            demo_sample_size = min(demo_size, len(self.demo_buffer))
            batch.extend(random.sample(list(self.demo_buffer), demo_sample_size))
        
        # 从交互缓冲区采样
        if len(self.interaction_buffer) > 0:
            interaction_sample_size = min(interaction_size, len(self.interaction_buffer))
            batch.extend(random.sample(list(self.interaction_buffer), interaction_sample_size))
        
        return batch
    
    def sample_interaction_batch(self, batch_size: int) -> List[Dict]:
        """仅从交互缓冲区采样"""
        if len(self.interaction_buffer) < batch_size:
            return list(self.interaction_buffer)
        return random.sample(list(self.interaction_buffer), batch_size)
    
    def get_refining_factor(self) -> float:
        """获取当前精炼因子"""
        if len(self.interaction_buffer) == 0:
            return float('inf')
        return len(self.demo_buffer) / len(self.interaction_buffer)
    
    def __len__(self):
        return len(self.demo_buffer) + len(self.interaction_buffer)