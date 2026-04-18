import torch
import numpy as np

class TaskGenerator:
    """
    任务生成器（Poisson 分布）
    对应 LaTeX Section III-B
    """
    
    def __init__(self, lambda_p: float = 0.5, num_task_types: int = 5):
        """
        参数:
            lambda_p: 单位时间平均任务生成数
            num_task_types: 任务类型数量
        """
        self.lambda_p = lambda_p
        self.num_task_types = num_task_types
    
    def generate_tasks(self, batch_size: int = 1) -> torch.Tensor:
        """
        生成任务到达（Poisson 分布）
        P(X=k) = (λ^k * e^(-λ)) / k!
        """
        tasks = torch.zeros(batch_size, self.num_task_types, 1)
        for b in range(batch_size):
            for i in range(self.num_task_types):
                k = np.random.poisson(self.lambda_p)
                tasks[b, i, 0] = min(k, 5)  # 限制最大任务数
        return tasks