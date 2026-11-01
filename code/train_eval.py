# -*- coding: utf-8 -*-
"""训练与评估：两种划分协议 × 多模型对比 + 外推预测。

协议:
  spatial  —— 按单元划分(70/15/15)，评估对未见单元的泛化
  temporal —— 按时间划分(train 2018-02~2018-11, val 2018-12~2019-02, test 2019-03~2019-04 窗口)

产物:
  outputs/results_<protocol>.csv      指标表
  outputs/pred_<model>_<protocol>.npz 测试预测
  outputs/history_*.json              训练曲线
  outputs/forecast.npz                全域外推预测(未来6/12个月)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import E_FORECAST, E_LAST, E_START, H_OUT, T_IN, make_splits
from model import build_model

DATA = r"E:\tianjindata\subsidence_project\data"
OUT = r"E:\tianjindata\subsidence_project\outputs"
os.makedirs(OUT, exist_ok=True)

D_SCALE = 20.0
HIDX = [5, 11]        # 6、12 个月（0-based）
W_H = torch.ones(H_OUT)
W_H[5] = 2.0
W_H[11] = 2.5

MODELS = ["stmagn", "stmagn_nodrv", "stmagn_nosp", "stmagn_nots",
          "mlp", "lstm", "transformer", "gcn", "gat"]


def sub_adj(A, idx):
    A2 = A[np.ix_(idx, idx)].clone()
    deg = A2.sum(1) + 1e-9
    return A2 / deg[:, None]


def train_one(name, protocol, seed=0, epochs=110, patience=18, lr=1e-3, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ud, tr, va, te, tr_e, va_e, te_e = make_splits(protocol, seed)
    ud.fit_scalers(tr)
    n_static = ud.static_raw.shape[1]

    A_all = torch.tensor(ud.A)
    A_tr = sub_adj(A_all, tr)
    A_va = sub_adj(A_all, va)
    A_te = sub_adj(A_all, te)

    model = build_model(name, n_static)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss(reduction="none")

    def build_batch(e_list, idx):
        """把多个窗口拼成一个批次: x_ts=[W*N,T], x_st=[N,F], x_drv=[W,T,C], y=[W*N,12]"""
        xs, xd, ys = [], [], []
        for e in e_list:
            a, b, c, d = ud.sample_batch(e, idx)
            xs.append(a); xd.append(c[None]); ys.append(d)
        x_ts = torch.tensor(np.concatenate(xs))
        x_st = torch.tensor(ud.static_norm()[idx])
        x_drv = torch.tensor(np.concatenate(xd))
        y = torch.tensor(np.concatenate(ys))
        return x_ts, x_st, x_drv, y

    batch_tr = build_batch(tr_e, tr)
    batch_va = build_batch(va_e, va)
    batch_te = build_batch(te_e, te)

    def run(batch, A, train=False):
        x_ts, x_st, x_drv, y = batch
        if train:
            opt.zero_grad()
            pred = model(x_ts, x_st, x_drv, A)
            loss = (loss_fn(pred, y) * W_H).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            with torch.no_grad():
                pred = model(x_ts, x_st, x_drv, A)
                loss = (loss_fn(pred, y) * W_H).mean()
        return float(loss.detach()), pred.detach()

    best = (1e9, None, 0)
    hist = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl, _ = run(batch_tr, A_tr, True)
        model.eval()
        vl, _ = run(batch_va, A_va, False)
        sched.step()
        hist["train"].append(tl); hist["val"].append(vl)
        if vl < best[0] - 1e-6:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= patience:
            break
        if verbose and ep % 10 == 0:
            print(f"   ep{ep:3d} train {tl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best[1])
    model.eval()

    # ---- 测试集预测（全部测试窗口） ----
    _, pred_t = run(batch_te, A_te, False)
    W = len(te_e)
    pred = pred_t.numpy().reshape(W, len(te), H_OUT) * D_SCALE
    obs = batch_te[3].numpy().reshape(W, len(te), H_OUT) * D_SCALE
    pred = pred.reshape(-1, H_OUT); obs = obs.reshape(-1, H_OUT)

    metrics = {}
    for hi, tag in zip(HIDX, ["h6", "h12"]):
        o, p = obs[:, hi], pred[:, hi]
        rmse = float(np.sqrt(np.mean((p - o) ** 2)))
        mae = float(np.mean(np.abs(p - o)))
        r2 = float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))
        r = float(np.corrcoef(p, o)[0, 1])
        sign = float(np.mean(np.sign(p) == np.sign(o)))
        metrics[tag] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
                        "R2": round(r2, 3), "Pearson_r": round(r, 4),
                        "趋势一致率": round(sign, 3)}
    return {"model": name, "protocol": protocol, "best_epoch": best[2],
            "val_loss": round(best[0], 4), **{f"{k}_RMSE": v["RMSE"] for k, v in metrics.items()},
            **{f"{k}_MAE": v["MAE"] for k, v in metrics.items()},
            **{f"{k}_R2": v["R2"] for k, v in metrics.items()},
            **{f"{k}_r": v["Pearson_r"] for k, v in metrics.items()},
            **{f"{k}_sign": v["趋势一致率"] for k, v in metrics.items()}}, pred, obs, hist, model, ud


def persistence_metrics(protocol):
    ud, tr, va, te, *_ , te_e = make_splits(protocol)
    y = []
    for e in te_e:
        y.append((ud.D[te, e + 1:e + 1 + H_OUT] - ud.D[te, e:e + 1]) / 1.0)
    obs = np.concatenate(y)
    pred = np.zeros_like(obs)
    out = {}
    for hi, tag in zip(HIDX, ["h6", "h12"]):
        o, p = obs[:, hi], pred[:, hi]
        out[tag] = {"RMSE": round(float(np.sqrt(np.mean((p - o) ** 2))), 2),
                    "MAE": round(float(np.mean(np.abs(p - o))), 2),
                    "R2": round(float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)), 3),
                    "Pearson_r": 0.0, "趋势一致率": round(float(np.mean(np.sign(p) == np.sign(o))), 3)}
    row = {"model": "persistence", "protocol": protocol, "best_epoch": -1, "val_loss": np.nan}
    row.update({f"{k}_RMSE": v["RMSE"] for k, v in out.items()})
    row.update({f"{k}_MAE": v["MAE"] for k, v in out.items()})
    row.update({f"{k}_R2": v["R2"] for k, v in out.items()})
    row.update({f"{k}_r": v["Pearson_r"] for k, v in out.items()})
    row.update({f"{k}_sign": v["趋势一致率"] for k, v in out.items()})
    return row


def train_final_and_forecast(epochs=130, seed=0):
    """用全部单元、全部窗口训练最终模型，并对最后一期外推 1~12 个月。"""
    ud = make_splits("spatial")[0]
    idx = np.arange(ud.n)
    ud.fit_scalers(idx)
    A = torch.tensor(ud.A)
    xs, xd, ys = [], [], []
    for e in range(E_START, E_LAST + 1):
        a, b, c, d = ud.sample_batch(e, idx)
        xs.append(a); xd.append(c[None]); ys.append(d)
    x_ts = torch.tensor(np.concatenate(xs))
    x_st = torch.tensor(ud.static_norm()[idx])
    x_drv = torch.tensor(np.concatenate(xd))
    y = torch.tensor(np.concatenate(ys))
    model = build_model("stmagn", ud.static_raw.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss(reduction="none")
    torch.manual_seed(seed)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_ts, x_st, x_drv, A)
        loss = (loss_fn(pred, y) * W_H).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        if ep % 10 == 0:
            print(f"   [final] ep{ep} loss {float(loss):.4f}", flush=True)
    model.eval()
    xf_ts, xf_st, xf_drv = ud.forecast_batch(E_FORECAST, idx)
    with torch.no_grad():
        pred = model(torch.tensor(xf_ts), torch.tensor(xf_st), torch.tensor(xf_drv), A).numpy()
    delta = pred * D_SCALE  # [N, 12] mm 增量
    last = ud.D[:, E_FORECAST][:, None]
    np.savez(os.path.join(OUT, "forecast.npz"), delta=delta, last_disp=last,
             base_ym=ud.ym[E_FORECAST], unit_idx=idx,
             x=ud.units.x.values, y=ud.units.y.values, cid=ud.units.cid.values,
             zone=ud.units.zone.values, vel=ud.units.vel.values)
    torch.save(model.state_dict(), os.path.join(OUT, "stmagn_final.pt"))
    print(f"[forecast] base={ud.ym[E_FORECAST]}  h6 mean={delta[:, 5].mean():.1f}mm  "
          f"h12 mean={delta[:, 11].mean():.1f}mm")
    return delta


def main():
    rows = []
    for protocol in ["spatial", "temporal"]:
        rows.append(persistence_metrics(protocol))
    logs = []
    for protocol in ["spatial", "temporal"]:
        for name in MODELS:
            t0 = time.time()
            row, pred, obs, hist, model, ud = train_one(name, protocol)
            rows.append(row)
            logs.append(row)
            np.savez(os.path.join(OUT, f"pred_{name}_{protocol}.npz"), pred=pred, obs=obs)
            with open(os.path.join(OUT, f"history_{name}_{protocol}.json"), "w") as f:
                json.dump(hist, f)
            print(f"[{protocol:8s}] {name:14s} h6 RMSE={row['h6_RMSE']:5.2f} "
                  f"MAE={row['h6_MAE']:5.2f} R2={row['h6_R2']:6.3f} r={row['h6_r']:6.3f} | "
                  f"h12 RMSE={row['h12_RMSE']:5.2f} R2={row['h12_R2']:6.3f}  "
                  f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results_all.csv"), index=False, encoding="utf-8")
    print(df.to_string(index=False))
    train_final_and_forecast()


if __name__ == "__main__":
    main()
