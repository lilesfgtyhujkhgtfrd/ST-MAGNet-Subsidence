# -*- coding: utf-8 -*-
"""可解释性分析:
  1) 门控融合权重 -> 空间/时序/诱发三分支贡献占比（分单元）
  2) Transformer 注意力 -> 关注的历史月份（热力图 + 峰值月份）
  3) GraphSAGE 邻居遮挡贡献度 -> 空间传播方向
  4) 诱发因子通道置换 -> 各驱动因子贡献与主导因子
  5) 静态环境因子置换 -> 环境因子重要性
  6) 沉降对地下水位响应的滞后时间（交叉相关）
产物: outputs/interpret.npz, outputs/tables/*.csv
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from dataset import E_FORECAST, STATIC_COLS, UnitData, H_OUT
from model import build_model

OUT = r"E:\tianjindata\subsidence_project\outputs"
DATA = r"E:\tianjindata\subsidence_project\data"
D_SCALE = 20.0
DRV_NAMES = ["降水", "气温", "地下水位", "地下水开采量", "土壤湿度"]
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)


def load_model():
    ud = UnitData()
    idx = np.arange(ud.n)
    ud.fit_scalers(idx)
    model = build_model("stmagn", ud.static_raw.shape[1])
    model.load_state_dict(torch.load(os.path.join(OUT, "stmagn_final.pt"), map_location="cpu"))
    model.eval()
    return ud, model, idx


def delta_pred(model, ud, e, idx, x_st=None, x_drv=None, need_attn=False):
    x_ts, x_st0, x_drv0 = ud.forecast_batch(e, idx)
    A = torch.tensor(ud.A)
    if x_st is None:
        x_st = x_st0
    if x_drv is None:
        x_drv = x_drv0
    with torch.no_grad():
        out = model(torch.tensor(x_ts), torch.tensor(x_st),
                    torch.tensor(x_drv), A, need_attn=need_attn)
    if need_attn:
        y, attn = out
        return y.numpy() * D_SCALE, attn.numpy()
    return out.numpy() * D_SCALE


def main():
    ud, model, idx = load_model()
    e = E_FORECAST
    base = delta_pred(model, ud, e, idx)          # [N,12]

    # ---------- 1) 门控权重 ----------
    g = model.fusion.last_gate.numpy()            # [N,160]
    gs, gt, gd = g[:, :64].mean(1), g[:, 64:128].mean(1), g[:, 128:].mean(1)
    tot = gs + gt + gd + 1e-9
    share = np.c_[gs / tot, gt / tot, gd / tot]

    # ---------- 2) 注意力 ----------
    _, attn = delta_pred(model, ud, e, idx, need_attn=True)   # [N, 12, 12]
    attn_last = attn[:, -1, :]                                 # 最后时刻对历史各月的注意力

    # ---------- 3) 邻居遮挡贡献 ----------
    x_st0 = ud.static_norm()
    A_mask = ud.A > 0
    vel = ud.units.vel.values
    top_targets = np.argsort(vel)[:25]             # 沉降最快(负值最小)的单元
    infl = {}
    for i in top_targets:
        nbrs = np.where(A_mask[i])[0]
        for j in nbrs:
            xs = x_st0.copy()
            xs[j] = 0.0                            # 标准化空间均值 = 0
            p = delta_pred(model, ud, e, idx, x_st=xs)[i, 11]
            infl[(int(i), int(j))] = float(p - base[i, 11])
    infl_df = pd.DataFrame([{"target": k[0], "neighbor": k[1], "delta_h12_mm": v}
                            for k, v in infl.items()])
    infl_df.to_csv(os.path.join(TAB, "neighbor_influence.csv"), index=False, encoding="utf-8")

    # 每个单元最强邻居（全单元计算：邻接加权 + 空间嵌入相似度近似）
    edge_i, edge_j = np.where(A_mask)
    # 以 h12 预测相对均值的偏离作为“沉降强度”，估计上游影响
    strength = base[:, 11]
    w = ud.A[edge_i, edge_j]
    score = w * np.abs(strength[edge_i])
    order = np.lexsort((edge_j, edge_i))
    edge_i, edge_j, score = edge_i[order], edge_j[order], score[order]
    top1 = pd.DataFrame({"target": edge_i, "neighbor": edge_j, "score": score})
    top1 = top1.sort_values("score", ascending=False).drop_duplicates("target")

    # ---------- 4) 诱发因子通道置换 ----------
    x_ts, x_st0b, x_drv0 = ud.forecast_batch(e, idx)
    drv_imp = np.zeros((len(DRV_NAMES), ud.n))
    for c in range(len(DRV_NAMES)):
        xd = x_drv0.copy()
        xd[:, c] = np.roll(xd[:, c], 5)            # 打乱时间顺序
        p = delta_pred(model, ud, e, idx, x_drv=xd)
        drv_imp[c] = np.abs(p[:, 11] - base[:, 11])
    drv_share = drv_imp / (drv_imp.sum(0, keepdims=True) + 1e-9)
    pd.DataFrame({"driver": DRV_NAMES, "mean_abs_delta_mm": drv_imp.mean(1).round(3)}
                 ).to_csv(os.path.join(TAB, "driver_importance.csv"), index=False, encoding="utf-8")

    # ---------- 5) 静态环境因子置换 ----------
    st_imp = np.zeros((len(STATIC_COLS), ud.n))
    for k in range(len(STATIC_COLS)):
        xs = x_st0.copy()
        xs[:, k] = 0.0
        p = delta_pred(model, ud, e, idx, x_st=xs)
        st_imp[k] = np.abs(p[:, 11] - base[:, 11])
    mean_imp = st_imp.mean(1)
    rank = np.empty(len(STATIC_COLS), dtype=int)
    rank[np.argsort(-mean_imp)] = np.arange(1, len(STATIC_COLS) + 1)
    pd.DataFrame({"feature": STATIC_COLS, "mean_abs_delta_mm": mean_imp.round(3), "rank": rank}
                 ).sort_values("rank").to_csv(
        os.path.join(TAB, "static_importance.csv"), index=False, encoding="utf-8")

    # ---------- 6) 滞后时间：单元位移变化 vs 地下水位 ----------
    drivers = pd.read_csv(os.path.join(DATA, "drivers.csv"))
    drivers = drivers.rename(columns={drivers.columns[0]: "ym"})
    drivers["ym"] = drivers["ym"].astype(str).str.split(".").str[0]
    drivers = drivers.set_index("ym")
    ym = ud.ym
    gw = drivers.reindex(ym)["gw_level_anom_m"].values.astype(float)
    gw = gw - gw.mean()
    D = ud.D
    rate = np.diff(D, axis=1)                      # 月增量
    lags = np.arange(0, 13)
    lag_best = np.zeros(ud.n, dtype=int)
    corr_best = np.zeros(ud.n)
    for i in range(ud.n):
        r = rate[i]
        r = r - r.mean()
        cs = []
        for L in lags:
            if L == 0:
                a, b = r, gw[1:]
            else:
                a = r[L:]
                b = gw[1:len(gw) - L] if len(gw) - L > 0 else gw[1:]
            n = min(len(a), len(b))
            if n < 10 or np.std(a[:n]) < 1e-6:
                cs.append(0.0); continue
            cs.append(float(np.corrcoef(a[:n], b[:n])[0, 1]))
        cs = np.array(cs)
        k = int(np.argmax(np.abs(cs)))
        lag_best[i], corr_best[i] = lags[k], cs[k]

    # ---------- 汇总落盘 ----------
    np.savez_compressed(os.path.join(OUT, "interpret.npz"),
                        gate_share=share, attn=attn, attn_last=attn_last,
                        driver_share=drv_share, driver_imp=drv_imp,
                        static_imp=st_imp, lag_best=lag_best, corr_best=corr_best,
                        base_delta=base, x=ud.units.x.values, y=ud.units.y.values,
                        cid=ud.units.cid.values, zone=ud.units.zone.values)
    np.savez(os.path.join(OUT, "edge_top1.npz"),
             target=top1.target.values, neighbor=top1.neighbor.values, score=top1.score.values)
    pd.DataFrame({"cid": ud.units.cid, "x": ud.units.x, "y": ud.units.y,
                  "zone": ud.units.zone_name, "vel": vel,
                  "gate_spatial": share[:, 0], "gate_temporal": share[:, 1], "gate_driver": share[:, 2],
                  "dom_driver": [DRV_NAMES[k] for k in np.argmax(drv_share, 0)],
                  "lag_month": lag_best, "lag_corr": corr_best,
                  "attn_peak_month": np.asarray(ud.ym)[e - 11 + np.argmax(attn_last, 1)],
                  "pred_h6_mm": base[:, 5], "pred_h12_mm": base[:, 11]}
                 ).to_csv(os.path.join(TAB, "unit_interpret.csv"), index=False, encoding="utf-8")

    print("[gate] 空间/时序/诱发 平均占比:", np.round(share.mean(0), 3))
    print("[driver] 通道重要性:", dict(zip(DRV_NAMES, np.round(drv_imp.mean(1), 3))))
    print("[static] top5:", list(pd.Series(st_imp.mean(1), index=STATIC_COLS).sort_values(ascending=False).head(5).index))
    print("[lag] 滞后中位数:", int(np.median(lag_best)), "月; 相关系数中位数:",
          round(float(np.median(corr_best)), 3))
    print("[save] outputs/interpret.npz & outputs/tables/*.csv")


if __name__ == "__main__":
    main()
