import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm
from data_generator import SceneDataGenerator
from model_unified import DTGATModel

def train_model(use_staff, scenario_id, epochs=100, lr=0.001, device='cuda'):
    print(f"--- Training DTGAT {'(with STAFF)' if use_staff else '(No STAFF)'} - Scenario {scenario_id} ---")
    os.makedirs(f"results/models/scenario_{scenario_id}", exist_ok=True)
    os.makedirs(f"results/logs", exist_ok=True)
    
    generator = SceneDataGenerator(batch_size=64)
    model = DTGATModel(use_staff=use_staff).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    best_loss = float('inf')
    converge_epoch = epochs
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _ in range(10): # 10 batches per epoch
            obs, hist, curr, adj, mask, label_imp, label_risk = generator.generate_batch()
            obs, hist, curr, adj, mask = obs.to(device), hist.to(device), curr.to(device), adj.to(device), mask.to(device)
            label_imp, label_risk = label_imp.to(device), label_risk.to(device)
            
            optimizer.zero_grad()
            pred_risk, pred_imp, _ = model(obs, hist, curr, adj, mask)
            
            # 仅计算有效节点的损失
            loss_imp = criterion(pred_imp * mask.unsqueeze(-1), label_imp * mask.unsqueeze(-1))
            loss_risk = criterion(pred_risk, label_risk)
            loss = loss_imp + loss_risk
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"results/models/scenario_{scenario_id}/model_{'staff' if use_staff else 'nostaff'}.pth")
        
        # 简单收敛判断 (连续 5 轮变化小于 1e-4)
        if epoch > 5 and abs(loss_history[-1] - loss_history[-2]) < 1e-4:
            if converge_epoch == epochs:
                converge_epoch = epoch + 1
    
    # 保存训练日志
    log_data = {
        "scenario": scenario_id,
        "use_staff": use_staff,
        "loss_history": loss_history,
        "converge_epoch": converge_epoch,
        "params": model.get_num_params()
    }
    with open(f"results/logs/log_scenario_{scenario_id}_{'staff' if use_staff else 'nostaff'}.json", 'w') as f:
        json.dump(log_data, f)
    
    print(f"Training Completed. Best Loss: {best_loss:.6f}, Converged at Epoch: {converge_epoch}")
    return log_data