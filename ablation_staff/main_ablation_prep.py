import os
from train_ablation import train_model
from evaluate_ablation import evaluate_model
from visualize_ablation import plot_training_curves, generate_latex_table

def main():
    print("######################################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    scenarios = [1, 2, 3, 4, 5] # 对应方案中的 S1-S5
    all_results = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for sid in scenarios:
        all_results[sid] = {}
        for use_staff in [True, False]:
            # 1. 训练
            log_data = train_model(use_staff, sid, epochs=300, device=device)
            
            # 2. 评估
            metrics = evaluate_model(use_staff, sid, device=device)
            metrics['converge_epoch'] = log_data['converge_epoch']
            all_results[sid]['staff' if use_staff else 'nostaff'] = metrics
            
            # 3. 可视化 (每个场景训练完后画一次)
            if not use_staff: # 只在第二轮画完对比图
                plot_training_curves(sid)
    
    # 4. 生成 LaTeX 表格
    generate_latex_table(all_results)
    print("=== Ablation Study Completed ===888######################################################")

if __name__ == "__main__":
    import torch
    main()