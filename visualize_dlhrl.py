import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_training_curves(log_path: str = 'results/dlhrl_logs/training_log.json'):
    """绘制训练曲线"""
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    episodes = [log['episode'] for log in logs]
    rewards = [log['avg_reward'] for log in logs]
    success_rates = [log['success_rate'] for log in logs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(episodes, rewards, 'r-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Reward Curve')
    ax1.grid(True)
    
    ax2.plot(episodes, success_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Curve')
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs('results/dlhrl_figures', exist_ok=True)
    plt.savefig('results/dlhrl_figures/training_curves.png', dpi=300)
    plt.show()

def plot_phi_evolution(log_path: str = 'results/dlhrl_logs/training_log.json'):
    """绘制奖励参数φ演化"""
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    episodes = [log['episode'] for log in logs]
    phi_values = [log['phi'] for log in logs]
    
    phi_array = np.array(phi_values)
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(episodes, phi_array[:, i], label=f'φ{i+1}', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward Weight φ')
    plt.title('DRMG Reward Parameter Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/dlhrl_figures/phi_evolution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
    plot_phi_evolution()