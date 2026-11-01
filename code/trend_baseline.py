# -*- coding: utf-8 -*-
"""趋势外推基线：用最近 12 个月的平均月增量线性外推，无训练。"""
import os

import numpy as np
import pandas as pd

import train_eval as T
from dataset import H_OUT, make_splits

RES = os.path.join(T.OUT, "results_all.csv")
HIDX = [5, 11]


def trend_metrics(protocol):
    ud, tr, va, te, tr_e, va_e, te_e = make_splits(protocol)
    obs, pred = [], []
    for e in te_e:
        v = (ud.D[te, e] - ud.D[te, e - 12]) / 12.0        # mm/月
        obs.append(ud.D[te, e + 1:e + 1 + H_OUT] - ud.D[te, e:e + 1])
        pred.append(np.outer(v, np.arange(1, H_OUT + 1)))
    obs = np.concatenate(obs); pred = np.concatenate(pred)
    row = {"model": "trend", "protocol": protocol, "best_epoch": -2, "val_loss": np.nan}
    for hi, tag in zip(HIDX, ["h6", "h12"]):
        o, p = obs[:, hi], pred[:, hi]
        row[f"{tag}_RMSE"] = round(float(np.sqrt(np.mean((p - o) ** 2))), 2)
        row[f"{tag}_MAE"] = round(float(np.mean(np.abs(p - o))), 2)
        row[f"{tag}_R2"] = round(float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)), 3)
        row[f"{tag}_r"] = round(float(np.corrcoef(p, o)[0, 1]), 4)
        row[f"{tag}_sign"] = round(float(np.mean(np.sign(p) == np.sign(o))), 3)
    return row


if __name__ == "__main__":
    rows = [trend_metrics(p) for p in ["spatial", "temporal"]]
    df = pd.read_csv(RES)
    df = df[df.model != "trend"]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(RES, index=False, encoding="utf-8")
    for r in rows:
        print(f"[trend] {r['protocol']:8s} h6 RMSE={r['h6_RMSE']} h12 RMSE={r['h12_RMSE']} "
              f"R2={r['h12_R2']} sign={r['h12_sign']}", flush=True)
