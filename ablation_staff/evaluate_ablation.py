import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import time
from data_generator import SceneDataGenerator
from model_unified import DTGATModel

def evaluate_model(use_staff, scenario_id, device='cuda'):
    print(f"--- Evaluating DTGAT {'(with STAFF)' if use_staff else '(No STAFF)'} - Scenario {scenario_id} ---")
    
    generator = SceneDataGenerator(batch_size=256)  # 大 batch 用于评估
    model = DTGATModel(use_staff=use_staff).to(device)
    model.load_state_dict(torch.load(f"results/models/scenario_{scenario_id}/model_{'staff' if use_staff else 'nostaff'}.pth"))
    model.eval()
    
    all_pred_imp, all_true_imp = [], []
    all_pred_risk, all_true_risk = [], []
    all_alphas = []
    infer_times = []
    
    with torch.no_grad():
        for _ in range(20):  # 20 test batches
            obs, hist, curr, adj, mask, label_imp, label_risk = generator.generate_batch()
            obs, hist, curr, adj, mask = obs.to(device), hist.to(device), curr.to(device), adj.to(device), mask.to(device)
            label_imp, label_risk = label_imp.to(device), label_risk.to(device)
            
            start = time.time()
            pred_risk, pred_imp, alpha = model(obs, hist, curr, adj, mask)
            end = time.time()
            infer_times.append((end - start) / obs.shape[0])  # 单样本时间
            
            # 【修复】应用 mask 收集有效数据（压缩最后一维）
            valid_mask = mask.bool()
            all_pred_imp.extend(pred_imp[valid_mask].squeeze(-1).cpu().numpy())
            all_true_imp.extend(label_imp[valid_mask].squeeze(-1).cpu().numpy())
            all_pred_risk.extend(pred_risk.squeeze(-1).cpu().numpy())  # 【修复】
            all_true_risk.extend(label_risk.squeeze(-1).cpu().numpy())  # 【修复】
            
            if use_staff and alpha is not None:
                all_alphas.extend(alpha[valid_mask].squeeze(-1).cpu().numpy())
    
    # 计算指标
    pred_imp_bin = (np.array(all_pred_imp) >= 0.5).astype(int)
    true_imp_bin = (np.array(all_true_imp) >= 0.5).astype(int)
    
    # 【修复】风险标签也需要二值化
    true_risk_bin = (np.array(all_true_risk) >= 0.5).astype(int)
    
    metrics = {
        "Acc_imp": accuracy_score(true_imp_bin, pred_imp_bin),
        "MSE_imp": mean_squared_error(all_true_imp, all_pred_imp),
        "AUC_imp": roc_auc_score(true_imp_bin, all_pred_imp),
        "AUC_risk": roc_auc_score(true_risk_bin, all_pred_risk),  # 【修复】使用二值化标签
        "T_infer": np.mean(infer_times) * 1000,  # ms
        "Params": model.get_num_params()
    }
    
    if use_staff and len(all_alphas) > 0:
        alpha_arr = np.array(all_alphas)
        # 计算熵 H(alpha) = -sum(p log p)
        eps = 1e-10
        h_alpha = -np.mean(alpha_arr * np.log(alpha_arr + eps) + (1-alpha_arr) * np.log(1-alpha_arr + eps))
        metrics["Alpha_Mean"] = np.mean(alpha_arr)
        metrics["Alpha_Entropy"] = h_alpha
        metrics["Alpha_Std"] = np.std(alpha_arr)
    else:
        metrics["Alpha_Mean"] = None
        metrics["Alpha_Entropy"] = None
        
    return metrics