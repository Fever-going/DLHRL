import torch
import torch.nn as nn
import torch.nn.functional as F
from STAFF import STAFFModule  # 确保 STAFF.py 在同目录

def normalize_adjacency(adj):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    deg = torch.sum(adj, dim=1)
    deg_inv = torch.pow(deg, -0.5)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    deg_inv = torch.diag(deg_inv)
    return torch.mm(deg_inv, torch.mm(adj, deg_inv))

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x, adj):
        support = self.linear(x)
        return torch.matmul(adj, support)

class GATConv(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, dropout=0.1):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x)
        h_i = h.unsqueeze(2)
        h_j = h.unsqueeze(1)
        concat_h = torch.cat([h_i.expand(-1, -1, N, -1), h_j.expand(-1, N, -1, -1)], dim=-1)
        e = torch.matmul(concat_h, self.a).squeeze(-1)
        e = self.leaky_relu(e)
        mask = (adj == 0).unsqueeze(0).expand(B, -1, -1)
        e.masked_fill_(mask, -1e9)
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, h)

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N, F, T)
        x = self.conv(x)
        x = x.view(B, N, -1, T).permute(0, 3, 1, 2).contiguous()
        return self.norm(x)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_risk_nodes=20, num_task_nodes=5):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_task_nodes = num_task_nodes
        self.obs_pos_fc = nn.Linear(input_dim, hidden_dim)
        self.obs_sem_fc = nn.Linear(input_dim, hidden_dim)
        self.state_fc = nn.Linear(input_dim, hidden_dim)
        self.task_node_proj = nn.Linear(hidden_dim, hidden_dim * num_task_nodes)

    def forward(self, obstacles, history_data, current_status):
        B = obstacles.shape[0]
        X_pos = F.relu(self.obs_pos_fc(obstacles))
        X_sem = F.relu(self.obs_sem_fc(obstacles))
        hist_feat = F.relu(self.state_fc(history_data))
        curr_feat = F.relu(self.state_fc(current_status))
        hist_proj = self.task_node_proj(hist_feat)
        X_history = hist_proj.view(B, -1, self.num_task_nodes, self.hidden_dim)
        curr_proj = self.task_node_proj(curr_feat)
        X_current = curr_proj.view(B, self.num_task_nodes, self.hidden_dim)
        return X_pos, X_sem, X_history, X_current

class RiskGraphBranch(nn.Module):
    def __init__(self, feature_dim=128, n_heads=8, window_size=5):
        super(RiskGraphBranch, self).__init__()
        self.window_size = window_size
        self.gat = GATConv(feature_dim, feature_dim, n_heads=1)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=n_heads, dropout=0.1, batch_first=False)
        self.mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))

    def forward(self, x_sem, adj_risk):
        B, T, N, F = x_sem.shape
        x_sem = x_sem.contiguous()
        x_flat = x_sem.view(B * T, N, F)
        gat_out = self.gat(x_flat, adj_risk)
        gat_out = gat_out.view(B, T, N, F).contiguous()
        x_perm = gat_out.permute(1, 0, 2, 3).contiguous().view(T, B * N, F)
        q = x_perm[-1:, :, :]
        attn_out, _ = self.temporal_attention(q, x_perm, x_perm)
        risk_feat = attn_out.squeeze(0).view(B, N, F).contiguous()
        return self.mlp(risk_feat)

class tGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(tGCNLayer, self).__init__()
        self.gcn = GCNConv(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        self.t_conv = TemporalConvBlock(out_features, out_features)
    def forward(self, x, adj):
        B, T, N, F = x.shape
        x_flat = x.contiguous().view(B * T, N, F)
        gc_out = self.gcn(x_flat, adj)
        gc_out = gc_out.view(B, T, N, -1).contiguous()
        gc_out = self.norm(gc_out)
        gc_out = self.relu(gc_out)
        return self.t_conv(gc_out)

class TaskGraphBranch(nn.Module):
    def __init__(self, feature_dim=128, use_staff=True):
        super(TaskGraphBranch, self).__init__()
        self.feature_dim = feature_dim
        self.use_staff = use_staff
        self.t_gcn_1 = tGCNLayer(feature_dim, feature_dim)
        self.t_gcn_2 = tGCNLayer(feature_dim * 2, feature_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.hist_proj = nn.Linear(3 * feature_dim, feature_dim)
        self.gcn_curr_1 = GCNConv(feature_dim, feature_dim)
        self.gcn_curr_2 = GCNConv(feature_dim, feature_dim)
        self.norm_curr = nn.LayerNorm(feature_dim)
        self.relu = nn.ReLU()
        
        if self.use_staff:
            self.staff_module = STAFFModule(in_dim=feature_dim, hidden_dim=64)
        else:
            self.fusion_fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x_history, x_current, adj_task, node_mask=None):
        B, T_hist, N, F = x_history.shape
        x_history = x_history.contiguous()
        x_current = x_current.contiguous()
        
        out1 = self.t_gcn_1(x_history, adj_task)
        concat1 = torch.cat([out1, x_history], dim=-1)
        out2 = self.t_gcn_2(concat1, adj_task)
        concat2 = torch.cat([out2, x_history, out1], dim=-1)
        concat2_perm = concat2.permute(0, 2, 3, 1).contiguous().view(B * N, -1, T_hist)
        pooled = self.pooling(concat2_perm).squeeze(-1)
        X_t_h = self.hist_proj(pooled.view(B, N, -1))
        
        gcn1_out = self.relu(self.norm_curr(self.gcn_curr_1(x_current, adj_task)))
        add1 = gcn1_out + x_current
        gcn2_out = self.relu(self.norm_curr(self.gcn_curr_2(add1, adj_task)))
        X_t_c = gcn2_out + add1
        
        if self.use_staff:
            task_feat, alpha = self.staff_module(X_t_h, X_t_c, adj_task)
        else:
            fused = torch.cat([X_t_h, X_t_c], dim=-1)
            task_feat = self.fusion_fc(fused)
            alpha = None
            
        if node_mask is not None:
            task_feat = task_feat * node_mask.unsqueeze(-1)
        return task_feat, alpha

class RiskHead(nn.Module):
    def __init__(self, feature_dim=128):
        super(RiskHead, self).__init__()
        self.head = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1), nn.Sigmoid())
    def forward(self, x):
        return self.head(x)

class ImportanceHead(nn.Module):
    def __init__(self, feature_dim=128):
        super(ImportanceHead, self).__init__()
        self.head = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1), nn.Sigmoid())
    def forward(self, x):
        return self.head(x)

class DTGATModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, window_size=5, n_heads=8,
                 num_nodes_risk=20, num_nodes_task=5, use_staff=True):
        super(DTGATModel, self).__init__()
        self.use_staff = use_staff
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, num_nodes_risk, num_nodes_task)
        self.risk_branch = RiskGraphBranch(hidden_dim, n_heads, window_size)
        self.task_branch = TaskGraphBranch(hidden_dim, use_staff=use_staff)
        self.risk_head = RiskHead(hidden_dim)
        self.importance_head = ImportanceHead(hidden_dim)
        self.num_nodes_risk = num_nodes_risk
        self.distance_threshold = 5.0

    def build_risk_adjacency(self, X_pos):
        B, N, D = X_pos.shape
        dist = torch.cdist(X_pos, X_pos, p=2)
        adj = (dist < self.distance_threshold).float()
        return normalize_adjacency(adj[0])

    def forward(self, obstacles, history_data, current_status, adj_task, node_mask=None):
        X_pos, X_sem, X_history, X_current = self.feature_extractor(obstacles, history_data, current_status)
        B, N, F = X_sem.shape
        T = self.risk_branch.window_size
        X_sem_seq = X_sem.unsqueeze(1).expand(-1, T, -1, -1).contiguous()
        adj_risk = self.build_risk_adjacency(X_pos)
        risk_feat = self.risk_branch(X_sem_seq, adj_risk)
        task_feat, alpha = self.task_branch(X_history, X_current, adj_task, node_mask)
        d_jt = self.risk_head(risk_feat)
        I_it = self.importance_head(task_feat)
        if node_mask is not None:
            I_it = I_it * node_mask.unsqueeze(-1)
        return d_jt, I_it, alpha

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)