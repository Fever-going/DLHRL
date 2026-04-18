import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_training_curves(scenario_id):
    plt.figure(figsize=(10, 6))
    for use_staff in [True, False]:
        log_file = f"results/logs/log_scenario_{scenario_id}_{'staff' if use_staff else 'nostaff'}.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            label = "DTGAT-Staff" if use_staff else "DTGAT-NoStaff"
            plt.plot(data['loss_history'], label=label)
            # 标注收敛点
            plt.axvline(x=data['converge_epoch'], linestyle='--', alpha=0.5)
    
    plt.title(f"Training Loss Comparison (Scenario {scenario_id})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/figures/loss_curve_s{scenario_id}.png")
    plt.close()

def plot_alpha_distribution(scenario_id):
    log_file = f"results/logs/log_scenario_{scenario_id}_staff.json"
    # 注意：alpha 分布需要在 evaluate 阶段保存，这里简化为从日志读取统计值或重新运行 evaluate 保存
    # 此处仅为示例结构，实际需修改 evaluate 脚本保存 alpha 直方图数据
    pass

def generate_latex_table(all_results):
    # all_results: dict { scenario: { staff: metrics, nostaff: metrics } }
    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study on STAFF Module}")
    lines.append("\\label{tab:ablation_staff}")
    lines.append("\\begin{tabular}{c|cc|cc|c}")
    lines.append("\\hline")
    lines.append("\\textbf{Scenario} & \\multicolumn{2}{c|}{\\textbf{Acc\\_imp}} & \\multicolumn{2}{c|}{\\textbf{MSE\\_imp}} & \\textbf{Conv.Epoch} \\\\")
    lines.append("& Staff & NoStaff & Staff & NoStaff & Gain \\\\")
    lines.append("\\hline")
    
    for sid, data in all_results.items():
        acc_s = f"{data['staff']['Acc_imp']:.4f}"
        acc_n = f"{data['nostaff']['Acc_imp']:.4f}"
        mse_s = f"{data['staff']['MSE_imp']:.4f}"
        mse_n = f"{data['nostaff']['MSE_imp']:.4f}"
        epoch_s = data['staff']['converge_epoch']
        epoch_n = data['nostaff']['converge_epoch']
        gain = ((epoch_n - epoch_s) / epoch_n) * 100
        
        lines.append(f"S{sid} & \\textbf{{{acc_s}}} & {acc_n} & \\textbf{{{mse_s}}} & {mse_n} & {gain:.1f}\\% \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open("results/ablation_table.tex", "w") as f:
        f.write("\n".join(lines))
    print("LaTeX table saved to results/ablation_table.tex")