import torch
import numpy as np
import os
import json
from dlhrl_core.dlhrl_agent import DLHRLAgent
from environment.grid_world import GridWorldEnv

def evaluate_dlhrl(model_path: str = 'results/dlhrl_models/dlhrl_agent.pth',
                   num_episodes: int = 100,
                   dtgat_model_path: str = 'ablation_staff/results/models/scenario_4/model_staff.pth'):
    """
    DLHRL 评估脚本
    对应 LaTeX Section VI-A Evaluation metrics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化
    env = GridWorldEnv(map_size=50, num_obstacles=20, num_tasks=5)
    agent = DLHRLAgent(dtgat_model_path=dtgat_model_path, device=device)
    agent.load_model(model_path)
    agent.actor_low.eval()
    
    # 评估指标
    metrics = {
        'NCT': [],  # 完成任务数量
        'APL': [],  # 平均路径长度
        'AEC': [],  # 平均能量消耗
        'WT': [],   # 任务等待时间
        'SR': []    # 成功率
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_steps = 0
        episode_energy = 0
        tasks_completed = 0
        wait_time = 0
        success = True
        
        while episode_steps < 200:
            task_data, trav_data = agent.get_stae_features(obs)
            state = {'position': obs['position'], 'power': obs['power']}
            action, is_safe = agent.select_action(state, task_data, trav_data, episode, explore=False)
            
            low_action = action[0, 2:]
            next_obs, reward, done, info = env.step(low_action[0])
            
            episode_steps += 1
            episode_energy += info['energy_consumption']
            if info['task_completed']:
                tasks_completed += 1
            if info['collision']:
                success = False
                break
            
            obs = next_obs
            
            if env._all_tasks_completed():
                break
        
        metrics['NCT'].append(tasks_completed)
        metrics['APL'].append(episode_steps)
        metrics['AEC'].append(episode_energy)
        metrics['WT'].append(wait_time)
        metrics['SR'].append(1.0 if success else 0.0)
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    print("\n=== DLHRL Evaluation Results ===")
    print(f"NCT (Completed Tasks): {avg_metrics['NCT']:.2f}")
    print(f"APL (Path Length): {avg_metrics['APL']:.2f}")
    print(f"AEC (Energy Consumption): {avg_metrics['AEC']:.2f}")
    print(f"WT (Waiting Time): {avg_metrics['WT']:.2f}")
    print(f"SR (Success Rate): {avg_metrics['SR']:.2%}")
    
    # 保存结果
    os.makedirs('results/dlhrl_logs', exist_ok=True)
    with open('results/dlhrl_logs/evaluation_results.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    return avg_metrics