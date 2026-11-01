# -*- coding: utf-8 -*-
"""数据容器: 滑窗样本、静态特征标准化、训练/验证/测试划分。"""
import json
import os

import numpy as np
import pandas as pd

DATA = r"E:\tianjindata\subsidence_project\data"

T_IN = 12          # 输入历史月数
H_OUT = 12         # 预测未来月数（取第 6、12 月作主指标）
D_SCALE = 20.0     # 位移标准化尺度(mm)
E_START = T_IN      # 首个可监督窗口末端索引(含)
E_LAST = 25         # 42 - 12 - 1
E_FORECAST = 37     # 最后一期，用于外推预测

STATIC_COLS = ["elev", "slope", "dist_water", "dist_major_road", "dist_road",
               "dist_rail", "dist_center", "building_density",
               "lu_urban", "lu_crop", "lu_water", "lu_other", "coh"]


class UnitData:
    def __init__(self):
        z = np.load(os.path.join(DATA, "units.npz"), allow_pickle=True)
        self.D = z["D"].astype(np.float32)                 # [N, 38] 累计垂直位移 mm
        self.Drv = z["Drv"].astype(np.float32)             # [38, C]
        self.units = pd.read_csv(os.path.join(DATA, "units.csv"))
        self.static_raw = self.units[STATIC_COLS].values.astype(np.float32)
        self.ym = z["ym"].tolist()
        self.n = self.D.shape[0]
        self.T = self.D.shape[1]
        g = np.load(os.path.join(DATA, "graph.npz"))
        self.A = g["A"].astype(np.float32)
        self.A_raw = g["A_raw"].astype(np.float32)
        self.edge_index = g["edge_index"]

    # ---- 标准化 ----
    def fit_scalers(self, train_idx):
        s = self.static_raw[train_idx]
        self.static_mean, self.static_std = s.mean(0), s.std(0) + 1e-6
        self.d_mean = self.D[train_idx].mean(0)
        self.d_std = self.D[train_idx].std(0) + 1e-6

    def static_norm(self):
        return (self.static_raw - self.static_mean) / self.static_std

    def sample_batch(self, e, idx):
        """返回第 e 个窗口末端、指定单元 idx 的 (x_ts, x_static, x_drv, y)。"""
        Dn = (self.D - self.d_mean) / self.d_std
        x_ts = Dn[idx, e - T_IN + 1:e + 1]                     # [n, T_IN] 标准化历史序列
        x_st = self.static_norm()[idx]
        x_drv = self.Drv[e - T_IN + 1:e + 1]                   # [T_IN, C]
        y = (self.D[idx, e + 1:e + 1 + H_OUT] - self.D[idx, e:e + 1]) / D_SCALE  # [n, 12]
        return x_ts, x_st, x_drv, y

    def forecast_batch(self, e, idx):
        Dn = (self.D - self.d_mean) / self.d_std
        x_ts = Dn[idx, e - T_IN + 1:e + 1]
        x_st = self.static_norm()[idx]
        x_drv = self.Drv[e - T_IN + 1:e + 1]
        return x_ts, x_st, x_drv

    def windows(self, e_range):
        return list(e_range)


def make_splits(protocol="spatial", seed=0):
    """返回 (train_units, val_units, test_units, train_e, val_e, test_e)。"""
    ud = UnitData()
    n = ud.n
    if protocol == "spatial":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_tr, n_va = int(0.7 * n), int(0.15 * n)
        tr = np.sort(perm[:n_tr]); va = np.sort(perm[n_tr:n_tr + n_va])
        te = np.sort(perm[n_tr + n_va:])
        e_all = list(range(E_START, E_LAST + 1))
        return ud, tr, va, te, e_all, [13, 18, 23], e_all
    elif protocol == "temporal":
        idx = np.arange(n)
        tr_e = list(range(E_START, 20))      # 11-19
        va_e = list(range(20, 23))           # 20-22
        te_e = list(range(23, E_LAST + 1))   # 23-25
        return ud, idx, idx, idx, tr_e, va_e, te_e
    else:
        raise ValueError(protocol)


if __name__ == "__main__":
    ud = UnitData()
    print("D", ud.D.shape, "Drv", ud.Drv.shape, "A", ud.A.shape, "ym", ud.ym[0], ud.ym[-1])
    with open(os.path.join(DATA, "units_meta.json"), encoding="utf-8") as f:
        m = json.load(f)
    print("units", m["n_units"], "epochs", m["n_epochs"])
