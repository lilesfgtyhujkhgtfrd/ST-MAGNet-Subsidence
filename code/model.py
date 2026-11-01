# -*- coding: utf-8 -*-
"""ST-MAGNet 三分支协同预测模型及对比基线。

三分支:
  空间分支  GraphSAGE  (dense 邻接聚合，均值聚合 + L2 归一化)
  时序分支  TransformerEncoder + 多周期位置编码（12/6/3 月周期）
  诱发分支  双层 GRU（地下水位/开采/降水/气温/土壤湿度）
融合:
  g = σ(W_g z_cat + b_g), z_res = g ⊙ z_cat + z_cat, z = LayerNorm(z_res)
  输出未来 12 个月逐月增量位移
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_IN = 12
H_OUT = 12
N_DRV = 5


# ---------------------------------------------------------------- 组件
class DenseGraphSAGE(nn.Module):
    """均值聚合 GraphSAGE（dense 邻接，适用于中等规模图）。"""

    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(nn.Linear(2 * d, hidden))
            d = hidden
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, A):
        h = x
        for lin in self.layers:
            nb = A @ h                      # 邻居均值
            h = F.relu(lin(torch.cat([h, nb], dim=-1)))
            h = F.normalize(h, p=2, dim=-1)
            h = F.dropout(h, self.dropout, self.training)
        return self.norm(h)


class MultiPeriodPE(nn.Module):
    """多周期位置编码：显式编码 12/6/3 月周期与线性趋势。"""

    def __init__(self, d_model, periods=(12.0, 6.0, 3.0), t_max=T_IN):
        super().__init__()
        pe = torch.zeros(t_max, 2 * len(periods) + 1)
        t = torch.arange(t_max, dtype=torch.float32)
        for k, p in enumerate(periods):
            pe[:, 2 * k] = torch.sin(2 * math.pi * t / p)
            pe[:, 2 * k + 1] = torch.cos(2 * math.pi * t / p)
        pe[:, -1] = t / t_max
        self.register_buffer("pe", pe)
        self.proj = nn.Linear(pe.shape[1], d_model)

    def forward(self, x):
        # x: [B, T, d] (同一单元共享位置编码)
        return x + self.proj(self.pe)[None, :, :]


class TemporalBranch(nn.Module):
    """Transformer 编码器时序分支，可返回注意力权重。"""

    def __init__(self, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pe = MultiPeriodPE(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.attn = None

    def forward(self, x, need_attn=False):
        h = self.pe(self.embed(x.unsqueeze(-1)) if x.dim() == 2 else x)
        # 逐层前向，记录最后一层注意力
        for i, layer in enumerate(self.encoder.layers):
            if need_attn and i == len(self.encoder.layers) - 1:
                _, a = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
                self.attn = a.detach()
            h = layer(h)
        return h[:, -1, :] if not need_attn else (h[:, -1, :], self.attn)


class DriverBranch(nn.Module):
    """诱发因子分支：双层 GRU。"""

    def __init__(self, in_dim=N_DRV, hidden=32, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out[:, -1, :])


class FusionHead(nn.Module):
    """门控融合 + 残差 + 层归一化 + 预测头。"""

    def __init__(self, d_cat, hidden=64, out_dim=H_OUT, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d_cat, d_cat)
        self.norm = nn.LayerNorm(d_cat)
        self.head = nn.Sequential(nn.Linear(d_cat, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.last_gate = None

    def forward(self, z):
        g = torch.sigmoid(self.gate(z))
        self.last_gate = g.detach()
        z = g * z + z
        z = self.norm(z)
        return self.head(z)


# ---------------------------------------------------------------- 主模型
class STMAGNet(nn.Module):
    def __init__(self, n_static, d_spatial=64, d_temporal=64, d_driver=32,
                 use_spatial=True, use_temporal=True, use_driver=True, dropout=0.1):
        super().__init__()
        self.use_spatial, self.use_temporal, self.use_driver = use_spatial, use_temporal, use_driver
        d_cat = 0
        if use_spatial:
            self.spatial = DenseGraphSAGE(n_static, d_spatial, 2, dropout)
            d_cat += d_spatial
        if use_temporal:
            self.temporal = TemporalBranch(d_temporal, 4, 2, dropout)
            d_cat += d_temporal
        if use_driver:
            self.driver = DriverBranch(N_DRV, d_driver, 2, dropout)
            d_cat += d_driver
        # 退化情况（仅时序/仅空间）不做门控，直接线性输出
        self.use_gate = use_spatial and use_temporal and use_driver
        self.fusion = FusionHead(d_cat, 64, H_OUT, dropout) if self.use_gate else \
            nn.Sequential(nn.Linear(d_cat, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, H_OUT))

    def encode(self, x_ts, x_st, x_drv, A, need_attn=False):
        """支持“多窗口批量”：x_ts 为 [W*N, T]，x_st 为 [N, F]，A 为 [N, N]。
        空间分支只在 N 个节点上计算一次后块状广播，避免逐窗口重复。"""
        B = x_ts.shape[0]
        parts = []
        attn = None
        if self.use_spatial:
            zs = self.spatial(x_st, A)
            if zs.shape[0] != B:
                zs = zs.repeat(B // zs.shape[0], 1)
            parts.append(zs)
        if self.use_temporal:
            if need_attn:
                zt, attn = self.temporal(x_ts, need_attn=True)
            else:
                zt = self.temporal(x_ts)
            parts.append(zt)
        if self.use_driver:
            xd = x_drv
            if xd.dim() == 2:                 # [T, C] -> 全节点共享
                zd = self.driver(xd.unsqueeze(0)).expand(B, -1)
            else:                             # [W, T, C] -> 逐窗口编码后展开到节点
                zd = self.driver(xd)
                if zd.shape[0] != B:
                    zd = zd.repeat_interleave(B // zd.shape[0], 0)
            parts.append(zd)
        z = torch.cat(parts, dim=-1)
        return z, attn

    def forward(self, x_ts, x_st, x_drv, A, need_attn=False):
        z, attn = self.encode(x_ts, x_st, x_drv, A, need_attn)
        y = self.fusion(z)
        if need_attn:
            return y, attn
        return y


# ---------------------------------------------------------------- 基线
class MLPBaseline(nn.Module):
    def __init__(self, n_static, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(T_IN + n_static, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, H_OUT))

    def forward(self, x_ts, x_st, x_drv, A, need_attn=False):
        if x_st.shape[0] != x_ts.shape[0]:
            x_st = x_st.repeat(x_ts.shape[0] // x_st.shape[0], 1)
        return self.net(torch.cat([x_ts, x_st], dim=-1))


class LSTMBaseline(nn.Module):
    """纯时序 LSTM（经典单点序列预测基线）。"""

    def __init__(self, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, H_OUT)

    def forward(self, x_ts, x_st, x_drv, A, need_attn=False):
        out, _ = self.lstm(x_ts.unsqueeze(-1))
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.tb = TemporalBranch(d_model, nhead, layers, dropout)
        self.head = nn.Linear(d_model, H_OUT)

    def forward(self, x_ts, x_st, x_drv, A, need_attn=False):
        z = self.tb(x_ts)
        return self.head(z)


class GCN(nn.Module):
    """Kipf GCN（dense，含对称归一化自环）。"""

    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim if i == 0 else hidden, hidden)
                                   for i in range(layers)])
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, A):
        A_sym = A + torch.eye(A.shape[0], device=A.device)
        d = A_sym.sum(1).clamp(min=1e-6)
        A_hat = A_sym / d.sqrt()[:, None] / d.sqrt()[None, :]
        h = x
        for lin in self.lins:
            h = F.relu(lin(A_hat @ h))
            h = F.dropout(h, self.dropout, self.training)
        return self.norm(h)


class GATLayer(nn.Module):
    """单头 GAT（基于稀疏边表，避免 n×n×d 的张量爆炸）。"""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = dropout

    def forward(self, h, A):
        n = h.shape[0]
        mask = A > 0
        ei = mask.nonzero().T                       # [2, E]
        self_loop = torch.arange(n, device=h.device)
        src = torch.cat([ei[0], self_loop]); dst = torch.cat([ei[1], self_loop])
        Wh = self.W(h)
        e = F.leaky_relu(self.a(torch.cat([Wh[src], Wh[dst]], -1)).squeeze(-1), 0.2)
        ew = torch.cat([A[ei[0], ei[1]], torch.ones(n, device=h.device)])
        e = e + torch.log(ew + 1e-6)
        # 按目标节点 scatter-softmax
        emax = torch.full((n,), -1e9, device=h.device).scatter_reduce(0, dst, e, reduce="amax")
        ex = torch.exp(e - emax[dst])
        den = torch.zeros(n, device=h.device).index_add_(0, dst, ex) + 1e-9
        alpha = ex / den[dst]
        alpha = F.dropout(alpha, self.dropout, self.training)
        out = torch.zeros_like(Wh).index_add_(0, dst, Wh[src] * alpha[:, None])
        return F.elu(out)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([GATLayer(in_dim if i == 0 else hidden, hidden, dropout)
                                    for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, A):
        h = x
        for l in self.layers:
            h = l(h, A)
        return self.norm(h)


class GNNLSTM(nn.Module):
    """空间(GCN/GAT) + 时序(LSTM) 双分支基线，无诱发因子分支。"""

    def __init__(self, n_static, kind="gcn", hidden=64, dropout=0.1):
        super().__init__()
        self.gnn = GCN(n_static, hidden, 2, dropout) if kind == "gcn" else GAT(n_static, hidden, 2, dropout)
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, H_OUT))

    def forward(self, x_ts, x_st, x_drv, A, need_attn=False):
        zs = self.gnn(x_st, A)
        if zs.shape[0] != x_ts.shape[0]:
            zs = zs.repeat(x_ts.shape[0] // zs.shape[0], 1)
        out, _ = self.lstm(x_ts.unsqueeze(-1))
        return self.head(torch.cat([zs, out[:, -1, :]], dim=-1))


def build_model(name, n_static):
    name = name.lower()
    if name == "stmagn":
        return STMAGNet(n_static)
    if name == "stmagn_nodrv":
        return STMAGNet(n_static, use_driver=False)
    if name == "stmagn_nosp":
        return STMAGNet(n_static, use_spatial=False)
    if name == "stmagn_nots":
        return STMAGNet(n_static, use_temporal=False)
    if name == "mlp":
        return MLPBaseline(n_static)
    if name == "lstm":
        return LSTMBaseline()
    if name == "transformer":
        return TransformerBaseline()
    if name == "gcn":
        return GNNLSTM(n_static, "gcn")
    if name == "gat":
        return GNNLSTM(n_static, "gat")
    raise ValueError(name)
