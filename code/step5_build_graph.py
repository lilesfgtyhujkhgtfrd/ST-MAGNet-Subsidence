# -*- coding: utf-8 -*-
"""Step5: 构建“空间邻近 + 环境相似”加权图结构。

节点 = 水文地质单元；边来自两类准则:
  1) 空间邻近: KNN(k=8) 最近邻
  2) 环境相似: 在最近 25 个候选中，环境因子余弦相似度最高的 4 个建立补充边
边权 w_ij = exp(-d_ij/d0) * (0.6 + 0.4 * sim_ij)，d0 取邻距中位数

产物: data/graph.npz
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DATA = r"E:\tianjindata\subsidence_project\data"

ENV_COLS = ["elev", "slope", "dist_water", "dist_road", "dist_major_road",
            "dist_rail", "building_density", "lu_urban", "lu_crop", "lu_water"]
K_SPATIAL = 8
K_CANDIDATE = 25
K_ENV = 4


def main():
    units = pd.read_csv(os.path.join(DATA, "units.csv"))
    xy = units[["x", "y"]].values.astype(np.float64)
    env = units[ENV_COLS].values.astype(np.float64)
    env = (env - env.mean(0)) / (env.std(0) + 1e-9)

    nn = NearestNeighbors(n_neighbors=K_CANDIDATE + 1, metric="euclidean").fit(xy)
    dist, idx = nn.kneighbors(xy)
    dist, idx = dist[:, 1:], idx[:, 1:]  # 去掉自身
    d0 = np.median(dist[:, 0]) * 1.5

    n = len(units)
    edges = {}
    # 环境余弦相似度（在候选集内）
    env_n = env / (np.linalg.norm(env, axis=1, keepdims=True) + 1e-9)
    for i in range(n):
        cand = idx[i]
        sim = env_n[cand] @ env_n[i]
        # 空间邻近边
        for j, d in zip(cand[:K_SPATIAL], dist[i, :K_SPATIAL]):
            w = np.exp(-d / d0) * (0.6 + 0.4 * (sim[cand.tolist().index(j)] + 1) / 2)
            edges[(i, j)] = max(edges.get((i, j), 0), w)
        # 环境相似边（相似度 > 0.6 的前 K_ENV 个）
        order = np.argsort(-sim)
        added = 0
        for k in order:
            if sim[k] < 0.6 or added >= K_ENV:
                break
            j, d = cand[k], dist[i, k]
            w = np.exp(-d / d0) * (0.5 + 0.5 * (sim[k] + 1) / 2)
            edges[(i, j)] = max(edges.get((i, j), 0), w)
            added += 1

    ei = np.array([[i, j] for (i, j) in edges], dtype=np.int64).T
    w = np.array(list(edges.values()), dtype=np.float32)
    # 对称化
    A = np.zeros((n, n), dtype=np.float32)
    A[ei[0], ei[1]] = w
    A = np.maximum(A, A.T)
    deg = A.sum(1) + 1e-9
    A_norm = A / deg[:, None]

    mean_k = int((A > 0).sum(1).mean())
    stats = {
        "n_nodes": n, "n_edges": int((A > 0).sum() // 2), "mean_degree": mean_k,
        "median_neighbor_dist_m": float(np.median(dist[:, 0])),
        "d0_m": float(d0), "K_SPATIAL": K_SPATIAL, "K_ENV": K_ENV,
        "env_cols": ENV_COLS,
    }
    np.savez_compressed(os.path.join(DATA, "graph.npz"), A=A_norm, A_raw=A,
                        edge_index=ei, weight=w)
    with open(os.path.join(DATA, "graph_meta.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    isolates = int((A.sum(1) == 0).sum())
    print(f"[graph] nodes={n} edges={stats['n_edges']} mean_degree={mean_k} "
          f"isolates={isolates} median_nn_dist={stats['median_neighbor_dist_m']:.0f}m")
    print("[save] data/graph.npz")


def _unused():  # placeholder removed
    pass


if __name__ == "__main__":
    main()
