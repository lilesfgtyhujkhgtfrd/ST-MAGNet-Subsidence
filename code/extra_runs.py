# -*- coding: utf-8 -*-
"""补充实验：
  1) 去时序分支消融的公平重跑（不做早停，跑满 110 epoch）
  2) ST-MAGNet 多随机种子稳定性（seed=1,2）
"""
import json
import os

import numpy as np
import pandas as pd

import train_eval as T

RES = os.path.join(T.OUT, "results_all.csv")
TAB = os.path.join(T.OUT, "tables")
os.makedirs(TAB, exist_ok=True)


def run_nots():
    rows = []
    for protocol in ["spatial", "temporal"]:
        row, pred, obs, hist, model, ud = T.train_one(
            "stmagn_nots", protocol, epochs=110, patience=200)
        rows.append(row)
        np.savez(os.path.join(T.OUT, f"pred_stmagn_nots_{protocol}.npz"), pred=pred, obs=obs)
        with open(os.path.join(T.OUT, f"history_stmagn_nots_{protocol}.json"), "w") as f:
            json.dump(hist, f)
    df = pd.read_csv(RES)
    df = df[df.model != "stmagn_nots"]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(RES, index=False, encoding="utf-8")
    print("[nots] re-run done", flush=True)


def run_stability():
    rows = []
    for protocol in ["spatial", "temporal"]:
        for seed in [1, 2]:
            r = T.train_one("stmagn", protocol, seed=seed, epochs=110, patience=200)[0]
            r["seed"] = seed
            rows.append(r)
            print(f"[stab] {protocol} seed{seed} h6={r['h6_RMSE']} h12={r['h12_RMSE']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "stmagn_stability.csv"),
                              index=False, encoding="utf-8")


if __name__ == "__main__":
    run_nots()
    run_stability()
