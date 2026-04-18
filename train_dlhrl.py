import torch
import numpy as np
import os
import json
from tqdm import tqdm
from dlhrl_core.dlhrl_agent import DLHRLAgent
from environment.grid_world import GridWorldEnv
from environment.task_generator import TaskGenerator

def train_dlhrl(num_episodes: int = 1500,
                max_steps: int = 200,
                batch_size: int = 256,
                dtgat_model_path: str = 'ablation_staff/results/models/scenario_4/model_staff.pth',
                save_dir: str = 'results/dlhrl_models',
                log_dir: str = 'results/dlhrl_logs'):
    """
    DLHRL 训练脚本
    对应 LaTeX Alg. 3
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化
    env = GridWorldEnv(map_size=50, num_obstacles=20, num_tasks=5)
    task_gen = TaskGenerator(lambda_p=0.5, num_task_types=5)
    agent = DLHRLAgent(dtgat_model_path=dtgat_model_path, device=device)
    
    # 训练统计
    episode_rewards = []
    success_rates = []
    training_logs = []
    
    # 训练循环
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        
        # 初始化队列
        #if agent.queue_state is None:
            #agent.queue_state = agent.lao.initialize_queues(1, agent.device)
        
        # 获取 STAE 特征
        # 获取 STAE 特征
        task_data, trav_data = agent.get_stae_features(obs)
        
        # 【调试】验证任务数据维度
        
        while step_count < max_steps:
            # 更新任务到达
            task_arrivals = task_gen.generate_tasks()
            task_data['task_arrivals'] = task_arrivals.to(agent.device)
            
            # 【修复】选择动作前确保状态张量在正确设备上
            state = {
                'position': obs['position'].to(agent.device),
                'power': obs['power'].to(agent.device)
            }
            
            # 【修复】确保 trav_data 中的张量也在正确设备上
            for key in trav_data:
                if isinstance(trav_data[key], torch.Tensor):
                    trav_data[key] = trav_data[key].to(agent.device)
            
            action, is_safe = agent.select_action(state, task_data, trav_data, episode)
            
            # 执行动作
            low_action = action[0, 2:]  # [v, θ]
            next_obs, reward, done, info = env.step(low_action)
            
            # 更新队列
            next_queue = agent.lao.update_queues(agent.queue_state, task_data)
            lyapunov_drift = agent.lao.compute_lyapunov_drift(agent.queue_state, next_queue)
            
            # 存储转移
            transition = {
                'state': state,
                'action': action,
                'reward': torch.tensor([[reward]]),
                'next_state': {
                    'position': next_obs['position'],
                    'power': next_obs['power']
                },
                'r_components': torch.zeros(4),
                'advantage': torch.zeros(1)
            }
            agent.store_transition(transition, is_safe)
            
            episode_reward += reward
            step_count += 1
            agent.step_count += 1
            
            # 更新网络
            if step_count % 40 == 0:
                critic_loss, actor_loss = agent.update_networks(batch_size)
                agent.update_drmg_params(lyapunov_drift)
            
            # 更新状态
            agent.queue_state = next_queue
            obs = next_obs
            task_data, trav_data = agent.get_stae_features(obs)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        success_rates.append(1.0 if not info['collision'] and env._all_tasks_completed() else 0.0)
        
        # 记录日志
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_sr = np.mean(success_rates[-50:])
            log_entry = {
                'episode': episode,
                'avg_reward': avg_reward,
                'success_rate': avg_sr,
                'phi': agent.drmg.get_phi().tolist()
            }
            training_logs.append(log_entry)
            print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, SR={avg_sr:.2f}")
    
    # 保存模型
    agent.save_model(os.path.join(save_dir, 'dlhrl_agent.pth'))
    
    # 保存日志
    with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    print(f"✓ Training completed. Model saved to {save_dir}")
    return training_logs