import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

class GridWorldEnv:
    """
    栅格世界环境
    对应 LaTeX Section III Problem Formulation
    """
    
    def __init__(self, map_size: int = 50,
                 num_obstacles: int = 20,
                 num_tasks: int = 5,
                 num_charging_stations: int = 2):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.num_tasks = num_tasks
        self.num_charging_stations = num_charging_stations
        
        self.reset()
    
    def reset(self) -> Dict:
        """重置环境"""
        # 机器人初始位置
        self.robot_pos = torch.tensor([np.random.uniform(0, self.map_size),
                                        np.random.uniform(0, self.map_size)])
        self.robot_battery = 100.0
        self.robot_velocity = 0.0
        self.robot_angle = 0.0
        
        # 生成障碍物
        self.obstacles = self._generate_obstacles()
        
        # 生成任务
        self.tasks = self._generate_tasks()
        
        # 生成充电站
        self.charging_stations = self._generate_charging_stations()
        
        # 生成地图
        self.P_map = self._generate_map()
        
        return self._get_observation()
    
    def _generate_obstacles(self) -> torch.Tensor:
        """生成障碍物"""
        obstacles = torch.zeros(1, self.num_obstacles, 64)  # [B=1, N, D]
        for i in range(self.num_obstacles):
            obstacles[0, i, 0] = np.random.uniform(0, self.map_size)  # x
            obstacles[0, i, 1] = np.random.uniform(0, self.map_size)  # y
            is_dynamic = np.random.rand() > 0.5
            if is_dynamic:
                obstacles[0, i, 2] = (np.random.rand() - 0.5) * 2.0  # vx
                obstacles[0, i, 3] = (np.random.rand() - 0.5) * 2.0  # vy
                obstacles[0, i, 4] = 1.0  # 动态
            else:
                obstacles[0, i, 4] = 0.0  # 静态
            obstacles[0, i, 5:] = torch.randn(59)  # 填充特征
        return obstacles
    
    def _generate_tasks(self) -> torch.Tensor:
        """生成任务"""
        tasks = []
        for i in range(self.num_tasks):
            task = {
                'position': torch.tensor([np.random.uniform(0, self.map_size),
                                          np.random.uniform(0, self.map_size)]),
                'completed': False,
                'energy_required': np.random.uniform(5, 15)
            }
            tasks.append(task)
        return tasks
    
    def _generate_charging_stations(self) -> torch.Tensor:
        """生成充电站"""
        stations = torch.zeros(self.num_charging_stations, 2)
        for i in range(self.num_charging_stations):
            stations[i, 0] = np.random.uniform(0, self.map_size)
            stations[i, 1] = np.random.uniform(0, self.map_size)
        return stations
    
    def _generate_map(self) -> torch.Tensor:
        """生成地图地形"""
        P_map = torch.ones(self.map_size, self.map_size) * 0.8
        # 添加一些障碍区域
        for _ in range(self.num_obstacles // 2):
            x = np.random.randint(0, self.map_size)
            y = np.random.randint(0, self.map_size)
            size = np.random.randint(3, 8)
            x_min = max(0, x - size)
            x_max = min(self.map_size, x + size)
            y_min = max(0, y - size)
            y_max = min(self.map_size, y + size)
            P_map[x_min:x_max, y_min:y_max] = 0.2
        return P_map
    
    def step(self, action: torch.Tensor) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        参数:
            action: [v, θ]
            
        返回:
            observation, reward, done, info
        """
        v = action[0].item()
        theta = action[1].item()
        
        # 更新机器人位置 (Eq. 1)
        dt = 0.1
        self.robot_pos[0] = self.robot_pos[0] + v * dt * np.sin(theta)
        self.robot_pos[1] = self.robot_pos[1] + v * dt * np.cos(theta)
        
        # 边界检查
        self.robot_pos = torch.clamp(self.robot_pos, 0, self.map_size)
        
        # 更新电量
        energy_consumption = 0.1 * v + 0.05
        self.robot_battery -= energy_consumption
        
        # 检查碰撞
        collision = self._check_collision()
        
        # 检查任务完成
        task_completed = self._check_task_completion()
        
        # 检查充电
        charging = self._check_charging()
        
        # 计算奖励
        reward = self._compute_reward(collision, task_completed, charging, energy_consumption)
        
        # 检查终止
        done = collision or self.robot_battery <= 0 or self._all_tasks_completed()
        
        info = {
            'collision': collision,
            'task_completed': task_completed,
            'charging': charging,
            'energy_consumption': energy_consumption
        }
        
        return self._get_observation(), reward, done, info
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        for i in range(self.num_obstacles):
            obs_pos = self.obstacles[0, i, :2]
            dist = torch.norm(self.robot_pos - obs_pos).item()
            if dist < 3.0:  # 安全距离
                return True
        return False
    
    def _check_task_completion(self) -> bool:
        """检查任务完成"""
        for task in self.tasks:
            if not task['completed']:
                dist = torch.norm(self.robot_pos - task['position']).item()
                if dist < 3.0:
                    task['completed'] = True
                    return True
        return False
    
    def _check_charging(self) -> bool:
        """检查充电"""
        for i in range(self.num_charging_stations):
            dist = torch.norm(self.robot_pos - self.charging_stations[i]).item()
            if dist < 3.0:
                self.robot_battery = min(100.0, self.robot_battery + 20.0)
                return True
        return False
    
    def _all_tasks_completed(self) -> bool:
        """检查所有任务是否完成"""
        return all(task['completed'] for task in self.tasks)
    
    def _compute_reward(self, collision: bool, task_completed: bool,
                        charging: bool, energy_consumption: float) -> float:
        """计算奖励"""
        reward = -energy_consumption  # r1
        
        if collision:
            reward -= 100.0
        if task_completed:
            reward += 10.0  # r3
        if charging:
            reward += 5.0
        
        return reward
    
    def _get_observation(self) -> Dict:
        """获取观测"""
        # 机器人历史
        history = torch.zeros(1, 5, 64)
        history[0, :, 0] = self.robot_pos[0]
        history[0, :, 1] = self.robot_pos[1]
        history[0, :, 2] = self.robot_battery / 100.0
        
        # 当前状态
        current = torch.zeros(1, 64)
        current[0, 0] = self.robot_pos[0]
        current[0, 1] = self.robot_pos[1]
        current[0, 2] = self.robot_battery / 100.0
        
        # 任务邻接矩阵
        adj_task = torch.ones(self.num_tasks, self.num_tasks) * 0.5
        adj_task = adj_task + torch.eye(self.num_tasks)
        deg = torch.sum(adj_task, dim=1)
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        deg_inv = torch.diag(deg_inv)
        adj_task = torch.mm(deg_inv, torch.mm(adj_task, deg_inv))
        
        return {
            'obstacles': self.obstacles,
            'robot_history': history,
            'robot_current': current,
            'task_adjacency': adj_task,
            'position': self.robot_pos.unsqueeze(0),
            'power': torch.tensor([[self.robot_battery / 100.0]]),
            'P_map': self.P_map,
            'obstacle_positions': self.obstacles[0, :, :2].unsqueeze(0)
        }
    
    def render(self):
        """可视化环境"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.P_map.numpy(), cmap='gray', origin='lower')
        
        # 机器人
        plt.plot(self.robot_pos[0].item(), self.robot_pos[1].item(), 'ro', markersize=15, label='Robot')
        
        # 任务
        for i, task in enumerate(self.tasks):
            color = 'green' if task['completed'] else 'yellow'
            plt.plot(task['position'][0].item(), task['position'][1].item(),
                    f'{color}o', markersize=10, label=f'Task {i+1}')
        
        # 充电站
        for i in range(self.num_charging_stations):
            plt.plot(self.charging_stations[i, 0].item(),
                    self.charging_stations[i, 1].item(),
                    'bs', markersize=12, label=f'Charging {i+1}')
        
        plt.legend()
        plt.title(f'Battery: {self.robot_battery:.1f}%')
        plt.grid(True)
        plt.show()