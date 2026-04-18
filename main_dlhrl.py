#!/usr/bin/env python3
"""
DLHRL 总控脚本
一键运行训练和评估流程
"""

import argparse
from train_dlhrl import train_dlhrl
from evaluate_dlhrl import evaluate_dlhrl

def main():
    parser = argparse.ArgumentParser(description='DLHRL Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'both'],
                        help='运行模式')
    parser.add_argument('--episodes', type=int, default=1500,
                        help='训练 episode 数量')
    parser.add_argument('--dtgat_path', type=str,
                        default='ablation_staff/results/models/scenario_4/model_staff.pth',
                        help='DTGAT 预训练权重路径')
    parser.add_argument('--save_dir', type=str,
                        default='results/dlhrl_models',
                        help='模型保存目录')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("Starting DLHRL Training")
        print("="*60)
        train_logs = train_dlhrl(
            num_episodes=args.episodes,
            dtgat_model_path=args.dtgat_path,
            save_dir=args.save_dir
        )
    
    if args.mode in ['evaluate', 'both']:
        print("\n" + "="*60)
        print("Starting DLHRL Evaluation")
        print("="*60)
        model_path = f"{args.save_dir}/dlhrl_agent.pth"
        metrics = evaluate_dlhrl(
            model_path=model_path,
            dtgat_model_path=args.dtgat_path
        )

if __name__ == "__main__":
    main()