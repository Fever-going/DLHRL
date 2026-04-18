import torch
import numpy as np

class SceneDataGenerator:
    def __init__(self, batch_size=32, num_risk_nodes=20, num_task_nodes=5, 
                 hist_length=5, feature_dim=64, map_size=100.0):
        self.batch_size = batch_size
        self.num_risk_nodes = num_risk_nodes
        self.num_task_nodes = num_task_nodes
        self.hist_length = hist_length
        self.feature_dim = feature_dim
        self.map_size = map_size

    def generate_obstacles(self):
        data = torch.zeros(self.batch_size, self.num_risk_nodes, self.feature_dim)
        for b in range(self.batch_size):
            for n in range(self.num_risk_nodes):
                data[b, n, 0] = torch.rand(1) * self.map_size  # x (索引 0)
                data[b, n, 1] = torch.rand(1) * self.map_size  # y (索引 1)
                is_dynamic = torch.rand(1) > 0.5
                if is_dynamic:
                    data[b, n, 2] = (torch.rand(1) - 0.5) * 2.0  # vx (索引 2)
                    data[b, n, 3] = (torch.rand(1) - 0.5) * 2.0  # vy (索引 3)
                    data[b, n, 4] = 1.0  # 类型：动态 (索引 4)
                else:
                    data[b, n, 4] = 0.0  # 类型：静态 (索引 4)
                data[b, n, 5] = torch.rand(1)  # 风险等级 (索引 5)
                # 【修复】索引 6 及以后填充随机噪声
                data[b, n, 6:] = torch.randn(self.feature_dim - 6)  # 【修复处 1】
        return data

    def generate_robot_history(self):
        data = torch.zeros(self.batch_size, self.hist_length, self.feature_dim)
        for b in range(self.batch_size):
            curr_x = torch.rand(1) * self.map_size
            curr_y = torch.rand(1) * self.map_size
            curr_batt = 80.0
            for t in range(self.hist_length):
                curr_x += (torch.rand(1) - 0.5) * 5.0
                curr_y += (torch.rand(1) - 0.5) * 5.0
                curr_x = torch.clamp(curr_x, 0, self.map_size)
                curr_y = torch.clamp(curr_y, 0, self.map_size)
                data[b, t, 0] = curr_x  # x (索引 0)
                data[b, t, 1] = curr_y  # y (索引 1)
                curr_batt -= torch.rand(1) * 2.0
                curr_batt = max(0, curr_batt)
                data[b, t, 2] = curr_batt / 100.0  # 电量 (索引 2)
                data[b, t, 3] = torch.rand(1) * 2.0  # 速度 (索引 3)
                data[b, t, 4] = torch.rand(1) * 2 * 3.14159  # 角度 (索引 4)
                # 【修复】索引 5 及以后填充随机噪声
                data[b, t, 5:] = torch.randn(self.feature_dim - 5)  # 【修复处 2】
        return data

    def generate_robot_current(self):
        data = torch.zeros(self.batch_size, self.feature_dim)
        for b in range(self.batch_size):
            data[b, 0] = torch.rand(1) * self.map_size  # x (索引 0)
            data[b, 1] = torch.rand(1) * self.map_size  # y (索引 1)
            data[b, 2] = torch.rand(1) * 100.0 / 100.0  # 电量 (索引 2)
            data[b, 3] = torch.rand(1) * 2.0  # 速度 (索引 3)
            data[b, 4] = torch.rand(1) * 2 * 3.14159  # 角度 (索引 4)
            # 【修复】索引 5 及以后填充随机噪声
            data[b, 5:] = torch.randn(self.feature_dim - 5)  # 【修复处 3】
        return data

    def generate_task_adjacency(self):
        task_pos = torch.rand(self.num_task_nodes, 2) * self.map_size
        dist = torch.cdist(task_pos, task_pos, p=2)
        threshold = 30.0
        adj = (dist < threshold).float()
        adj = adj + torch.eye(self.num_task_nodes)
        deg = torch.sum(adj, dim=1)
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def generate_node_mask(self):
        mask = torch.ones(self.batch_size, self.num_task_nodes)
        for b in range(self.batch_size):
            valid_count = torch.randint(2, self.num_task_nodes + 1, (1,)).item()
            mask[b, valid_count:] = 0
        return mask

    def generate_ground_truth(self, obstacles, current_status, task_adj):
        B = self.batch_size
        N_task = self.num_task_nodes
        N_risk = self.num_risk_nodes
        
        # 1. 任务重要性标签
        task_positions = torch.rand(B, N_task, 2) * self.map_size
        robot_pos = current_status[:, :2].unsqueeze(1).expand(-1, N_task, -1)
        dist_to_task = torch.norm(task_positions - robot_pos, p=2, dim=-1)
        battery = current_status[:, 2].unsqueeze(1).expand(-1, N_task)
        norm_dist = 1.0 - (dist_to_task / (self.map_size * 1.414))
        urgency = (1.0 - battery) * 0.5 + norm_dist * 0.5
        label_importance = torch.clamp(urgency, 0.0, 1.0).unsqueeze(-1)
        
        # 2. 碰撞风险标签
        obs_pos = obstacles[:, :, :2]
        robot_pos_exp = robot_pos[:, :1, :].expand(-1, N_risk, -1)
        dist_to_obs = torch.norm(obs_pos - robot_pos_exp, p=2, dim=-1)
        label_risk = torch.exp(-dist_to_obs / 10.0).unsqueeze(-1)
        
        return label_importance, label_risk

    def generate_batch(self):
        obstacles = self.generate_obstacles()
        history = self.generate_robot_history()
        current = self.generate_robot_current()
        adj_task = self.generate_task_adjacency()
        node_mask = self.generate_node_mask()
        label_imp, label_risk = self.generate_ground_truth(obstacles, current, adj_task)
        return obstacles, history, current, adj_task, node_mask, label_imp, label_risk