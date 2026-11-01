# -*- coding: utf-8 -*-
"""成果图件生成。"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\tianjindata\subsidence_project"
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)

C_MAIN, C_RED, C_BLUE, C_GRAY = "#1f4e79", "#c0392b", "#2980b9", "#7f8c8d"


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", name)


def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def main():
    units = pd.read_csv(os.path.join(DATA, "units.csv"))
    ps = pd.read_parquet(os.path.join(DATA, "ps_points.parquet"),
                         columns=["lon", "lat", "vel", "coh"])
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    u = np.load(os.path.join(DATA, "units.npz"), allow_pickle=True)
    D = u["D"]; ym = u["ym"].tolist(); epochs = u["epochs"]
    fc = np.load(os.path.join(OUT, "forecast.npz"))
    it = np.load(os.path.join(OUT, "interpret.npz"))
    res = pd.read_csv(os.path.join(OUT, "results_all.csv"))

    # ---------------- F01 PS 点平均沉降速率 ----------------
    s = ps.sample(n=min(120000, len(ps)), random_state=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(s.lon, s.lat, c=s.vel, s=0.6, cmap="RdYlBu_r", vmin=-50, vmax=10,
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("LOS 年均形变速率 (mm/a)")
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title(f"研究区 PS-InSAR 年均形变速率（{len(ps):,} 个永久散射体，2017-03—2020-03）", fontsize=11)
    ax.grid(alpha=0.2, ls=":")
    save(fig, "F01_ps_velocity_map.png")

    # ---------------- F02 单元划分与分区 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    sc = ax.scatter(units.x / 1000, units.y / 1000, c=units.vel, s=18,
                    cmap="RdYlBu_r", vmin=-50, vmax=10, edgecolors="none")
    fig.colorbar(sc, ax=ax, shrink=0.8).set_label("单元年均沉降速率 (mm/a)")
    strip_axes(ax); ax.set_title(f"水文地质单元（{len(units)} 个，600 m 网格）")
    ax = axes[1]
    zcolors = ["#2c7fb8", "#41b6c4", "#fecc5c", "#e34a33"]
    for k, zn in enumerate(sorted(units.zone_name.unique())):
        g = units[units.zone_name == zn]
        ax.scatter(g.x / 1000, g.y / 1000, s=18, c=zcolors[k % 4], label=f"{zn} (n={len(g)})",
                   edgecolors="none")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    strip_axes(ax); ax.set_title("地理亚区划分（KMeans, k=4）")
    save(fig, "F02_units_and_zones.png")

    # ---------------- F03 区域时序与驱动 ----------------
    drivers = pd.read_csv(os.path.join(DATA, "drivers.csv"))
    drivers = drivers.rename(columns={drivers.columns[0]: "ym"})
    drivers["ym"] = drivers["ym"].astype(str).str.split(".").str[0]
    drivers = drivers.set_index("ym").loc[ym]
    t = np.arange(len(ym))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, zn in enumerate(sorted(units.zone_name.unique())):
        zi = units.index[units.zone_name == zn]
        ax.plot(t, D[zi].mean(0), lw=1.8, color=zcolors[k % 4], label=zn)
    ax.set_xticks(t[::3]); ax.set_xticklabels([ym[i] for i in range(0, len(ym), 3)], rotation=45, fontsize=8)
    ax.set_ylabel("累计垂直位移 (mm)"); ax.axhline(0, color="k", lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(t, drivers["gw_level_anom_m"].values, color=C_RED, lw=1.4, ls="--", label="深层地下水位异常")
    ax2.set_ylabel("深层地下水位异常 (m)", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title("各亚区累计沉降过程线与深层地下水位变化（2017-03—2020-03）", fontsize=11)
    ax.grid(alpha=0.2, ls=":")
    save(fig, "F03_regional_series.png")

    # ---------------- F04 典型单元时序 ----------------
    order = np.argsort(units.vel.values)
    picks = [order[0], order[len(order) // 6], order[len(order) // 2], order[-1]]
    names = ["快速沉降单元", "中等沉降单元", "缓慢沉降单元", "相对抬升单元"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, p, nm in zip(axes.ravel(), picks, names):
        ax.plot(t, D[p], "-o", ms=2.4, lw=1.4, color=C_MAIN)
        coef = np.polyfit(t, D[p], 1)
        ax.plot(t, np.polyval(coef, t), ls="--", color=C_RED, lw=1.2,
                label=f"线性趋势 {coef[0]*12:.1f} mm/a")
        ax.set_title(f"{nm} #{units.cid[p]}（{ym[0]}—{ym[-1]}）", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.2, ls=":")
        ax.set_ylabel("垂直位移 (mm)")
    for ax in axes[1]:
        ax.set_xticks(t[::6]); ax.set_xticklabels([ym[i] for i in range(0, len(ym), 6)], rotation=30, fontsize=8)
    save(fig, "F04_typical_units.png")

    # ---------------- F05 训练曲线 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, proto in zip(axes, ["spatial", "temporal"]):
        h = json.load(open(os.path.join(OUT, f"history_stmagn_{proto}.json")))
        ax.plot(h["train"], label="训练", color=C_BLUE)
        ax.plot(h["val"], label="验证", color=C_RED)
        ax.set_title(f"ST-MAGNet 收敛曲线（{ '空间划分' if proto=='spatial' else '时间划分'}）", fontsize=10)
        ax.set_xlabel("epoch"); ax.set_ylabel("加权 MSE"); ax.legend(fontsize=8); ax.grid(alpha=0.2, ls=":")
    save(fig, "F05_training_curves.png")

    # ---------------- F06 模型对比 ----------------
    label_map = {"persistence": "持续法", "trend": "趋势外推法", "stmagn": "ST-MAGNet（本项目）",
                 "stmagn_nodrv": "ST-MAGNet 去诱发分支",
                 "stmagn_nosp": "ST-MAGNet 去空间分支", "stmagn_nots": "ST-MAGNet 去时序分支",
                 "mlp": "MLP", "lstm": "LSTM",
                 "transformer": "Transformer", "gcn": "GCN+LSTM", "gat": "GAT+LSTM"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for ax, proto, title in zip(axes, ["spatial", "temporal"], ["未见单元泛化", "时间外推"]):
        d = res[res.protocol == proto].copy()
        d["label"] = d.model.map(label_map)
        d = d.sort_values("h6_RMSE")
        y = np.arange(len(d))
        ax.barh(y - 0.2, d.h6_RMSE, height=0.38, color=C_BLUE, label="6 个月")
        ax.barh(y + 0.2, d.h12_RMSE, height=0.38, color=C_RED, label="12 个月")
        for yy, v in zip(y - 0.2, d.h6_RMSE):
            ax.text(v + 0.15, yy, f"{v:.1f}", va="center", fontsize=7)
        for yy, v in zip(y + 0.2, d.h12_RMSE):
            ax.text(v + 0.15, yy, f"{v:.1f}", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(d.h12_RMSE.max(), d.h6_RMSE.max()) * 1.13)
        ax.set_xlabel("RMSE (mm)"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.2, ls=":")
    save(fig, "F06_model_comparison.png")

    # ---------------- F07 预测-实测散点 ----------------
    pz = np.load(os.path.join(OUT, "pred_stmagn_spatial.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, hi, nm in zip(axes, [5, 11], ["未来 6 个月", "未来 12 个月"]):
        o, p = pz["obs"][:, hi], pz["pred"][:, hi]
        ax.scatter(o, p, s=6, alpha=0.45, color=C_BLUE, linewidths=0)
        lim = [min(o.min(), p.min()) - 2, max(o.max(), p.max()) + 2]
        ax.plot(lim, lim, "k--", lw=1)
        rmse = np.sqrt(np.mean((p - o) ** 2))
        r2 = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("InSAR 实测累计位移 (mm)"); ax.set_ylabel("模型预测值 (mm)")
        ax.set_title(f"{nm}（RMSE={rmse:.1f} mm, R²={r2:.2f}）", fontsize=10)
        ax.grid(alpha=0.2, ls=":")
    save(fig, "F07_pred_vs_obs.png")

    # ---------------- F08 外推预测分布 ----------------
    def add_months(ym_str, h):
        y, m = int(ym_str[:4]), int(ym_str[4:6])
        m2 = m - 1 + h
        return f"{y + m2 // 12}-{m2 % 12 + 1:02d}"

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5))
    for ax, hi, nm in zip(axes, [5, 11], ["6 个月", "12 个月"]):
        # 累计位移 = 末期实测 + 预测增量
        cum = fc["last_disp"][:, 0] + fc["delta"][:, hi]
        sc = ax.scatter(fc["x"] / 1000, fc["y"] / 1000, c=cum, s=20, cmap="YlGnBu_r",
                        vmin=np.percentile(cum, 2), vmax=np.percentile(cum, 98), edgecolors="none")
        fig.colorbar(sc, ax=ax, shrink=0.8).set_label("预测累计垂直位移 (mm)")
        strip_axes(ax)
        ax.set_title(f"未来 {nm}（{add_months(str(fc['base_ym']), hi + 1)}）", fontsize=10)
    save(fig, "F08_forecast_maps.png")

    # ---------------- F09 注意力 ----------------
    attn = it["attn_last"]
    top = np.argsort(units.vel.values)[:60]
    peak = attn[top].sum(1)
    sel = top[np.argsort(-peak)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    im = axes[0].imshow(attn[sel], aspect="auto", cmap="YlOrRd")
    axes[0].set_yticks(range(len(sel))[::10])
    axes[0].set_yticklabels([f"#{units.cid[i]}" for i in sel][::10], fontsize=8)
    axes[0].set_xticks(range(12)); axes[0].set_xticklabels(ym[-12:], rotation=45, fontsize=8)
    axes[0].set_ylabel("沉降最快前 60 个单元"); axes[0].set_title("Transformer 最后一时刻对历史月份的注意力", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].bar(range(12), attn.mean(0), color=C_MAIN)
    axes[1].set_xticks(range(12)); axes[1].set_xticklabels(ym[-12:], rotation=45, fontsize=8)
    axes[1].set_title("全区域平均注意力", fontsize=10); axes[1].grid(axis="y", alpha=0.2, ls=":")
    save(fig, "F09_attention.png")

    # ---------------- F10 空间传播 ----------------
    ed = np.load(os.path.join(OUT, "edge_top1.npz"))
    targets = ed["target"]; nbrs = ed["neighbor"]
    base = it["base_delta"][:, 11]
    strong = np.argsort(base)[:30]
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.scatter(units.x / 1000, units.y / 1000, c=units.vel, s=16, cmap="RdYlBu_r",
               vmin=-50, vmax=10, edgecolors="none", alpha=0.85)
    for i in strong:
        j = nbrs[np.where(targets == i)[0][0]]
        ax.annotate("", xy=(units.x[i] / 1000, units.y[i] / 1000),
                    xytext=(units.x[j] / 1000, units.y[j] / 1000),
                    arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.2, alpha=0.85))
        ax.scatter(units.x[i] / 1000, units.y[i] / 1000, s=40, facecolors="none", edgecolors="k", lw=0.6)
    strip_axes(ax); ax.set_title("沉降空间传播路径估计（箭头：最强邻域贡献方向）", fontsize=10)
    save(fig, "F10_spatial_propagation.png")

    # ---------------- F11 门控分支贡献 ----------------
    share = it["gate_share"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for k, (nm, cm) in enumerate(zip(["空间分支", "时序分支", "诱发因子分支"], ["Blues", "Greens", "Oranges"])):
        sc = axes[k].scatter(units.x / 1000, units.y / 1000, c=share[:, k], s=16, cmap=cm,
                             edgecolors="none")
        fig.colorbar(sc, ax=axes[k], shrink=0.78).set_label("门控权重占比")
        strip_axes(axes[k]); axes[k].set_title(nm, fontsize=10)
    axes[3].bar(["空间", "时序", "诱发"], share.mean(0), color=[C_BLUE, "#27ae60", "#e67e22"])
    axes[3].set_ylim(0, 0.6); axes[3].set_title("全区域平均贡献占比", fontsize=10)
    axes[3].grid(axis="y", alpha=0.2, ls=":")
    save(fig, "F11_gate_contribution.png")

    # ---------------- F12 主导驱动因子 ----------------
    dsh = it["driver_share"]
    dom = np.argmax(dsh, 0)
    di = pd.read_csv(os.path.join(OUT, "tables", "driver_importance.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.8, 1]})
    cols = ["#2980b9", "#c0392b", "#8e44ad", "#e67e22", "#16a085"]
    gwshare = dsh[2] + dsh[3]          # 地下水位 + 开采量
    sc = axes[0].scatter(units.x / 1000, units.y / 1000, c=gwshare, s=18, cmap="YlOrRd",
                         edgecolors="none", vmin=max(gwshare.min(), 0.0))
    fig.colorbar(sc, ax=axes[0], shrink=0.8).set_label("地下水位 + 开采量 通道占比")
    strip_axes(axes[0])
    axes[0].set_title(f"各单元地下水相关驱动占比（均值 {gwshare.mean():.2f}）", fontsize=10)
    axes[1].barh(di.driver, di.mean_abs_delta_mm, color=cols)
    axes[1].invert_yaxis(); axes[1].set_xlabel("置换后 |Δ预测| (mm)")
    axes[1].set_title("诱发因子通道重要性", fontsize=10); axes[1].grid(axis="x", alpha=0.2, ls=":")
    save(fig, "F12_driver_dominance.png")

    # ---------------- F13 滞后 ----------------
    lag, lc = it["lag_best"], it["corr_best"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    sc = axes[0].scatter(units.x / 1000, units.y / 1000, c=lag, s=18, cmap="viridis", edgecolors="none")
    fig.colorbar(sc, ax=axes[0], shrink=0.8).set_label("最优滞后 (月)")
    strip_axes(axes[0]); axes[0].set_title("沉降对地下水位响应的滞后时间", fontsize=10)
    axes[1].hist(lag, bins=np.arange(-0.5, 13.5, 1), color=C_MAIN, rwidth=0.85)
    axes[1].set_xlabel("滞后 (月)"); axes[1].set_ylabel("单元数")
    axes[1].set_title(f"滞后分布（中位数 {int(np.median(lag))} 月）", fontsize=10)
    axes[1].grid(axis="y", alpha=0.2, ls=":")
    save(fig, "F13_lag_response.png")

    # ---------------- F14 亚区对比 ----------------
    zs = units.groupby("zone_name").agg(vel=("vel", "mean"), n=("cid", "size")).reset_index()
    fcu = pd.DataFrame({"zone_name": units.zone_name, "h6": fc["delta"][:, 5], "h12": fc["delta"][:, 11]})
    zs = zs.merge(fcu.groupby("zone_name").mean().reset_index(), on="zone_name")
    zs = zs.merge(pd.DataFrame({"zone_name": units.zone_name,
                                **{f"g{k}": share[:, k] for k in range(3)}})
                  .groupby("zone_name").mean().reset_index(), on="zone_name")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4))
    x = np.arange(len(zs)); wd = 0.26
    axes[0].bar(x - wd, zs.vel * 1.0, wd, color=C_BLUE, label="历史年均速率")
    axes[0].bar(x, zs.h6, wd, color="#e67e22", label="预测 6 个月增量")
    axes[0].bar(x + wd, zs.h12, wd, color=C_RED, label="预测 12 个月增量")
    axes[0].set_xticks(x); axes[0].set_xticklabels(zs.zone_name, fontsize=8)
    axes[0].set_ylabel("mm 或 mm/a"); axes[0].legend(fontsize=7); axes[0].set_title("亚区沉降量与预测", fontsize=10)
    axes[0].grid(axis="y", alpha=0.2, ls=":")
    bottom = np.zeros(len(zs))
    for k, (nm, cm) in enumerate(zip(["空间", "时序", "诱发"], [C_BLUE, "#27ae60", "#e67e22"])):
        axes[1].bar(x, zs[f"g{k}"], bottom=bottom, color=cm, label=nm)
        bottom += zs[f"g{k}"].values
    axes[1].set_xticks(x); axes[1].set_xticklabels(zs.zone_name, fontsize=8)
    axes[1].set_title("亚区三分支贡献构成", fontsize=10); axes[1].legend(fontsize=8)
    dsh_z = pd.DataFrame({"zone": units.zone_name,
                          **{nm: dsh[k] for k, nm in enumerate(["降水", "气温", "地下水位", "开采量", "土壤湿度"])}}
                         ).groupby("zone").mean().reindex(zs.zone_name)
    bottom = np.zeros(len(dsh_z))
    for k, nm in enumerate(dsh_z.columns):
        axes[2].bar(x, dsh_z[nm], 0.6, bottom=bottom, color=cols[k], label=nm)
        bottom += dsh_z[nm].values
    axes[2].set_xticks(x); axes[2].set_xticklabels(zs.zone_name, fontsize=8)
    axes[2].set_ylabel("通道占比"); axes[2].legend(fontsize=6.5, ncol=5, loc="upper center",
                                                 bbox_to_anchor=(0.5, -0.12), frameon=False)
    axes[2].set_title("亚区诱发通道贡献构成", fontsize=10); axes[2].grid(axis="y", alpha=0.2, ls=":")
    save(fig, "F14_zone_analysis.png")

    # ---------------- F15 未来 12 个月累计曲线 ----------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    dlt = fc["delta"]
    for i in np.argsort(fc["last_disp"][:, 0] + dlt[:, 11])[:10]:
        ax.plot(np.arange(1, 13), fc["last_disp"][i, 0] + dlt[i], lw=1.2, alpha=0.85,
                label=f"#{fc['cid'][i]} ({dlt[i, 11]:.0f} mm)")
    ax.set_xlabel("预测月份（自 2020-03 起）"); ax.set_ylabel("累计垂直位移 (mm)")
    ax.set_title("沉降最严重 10 个单元的 12 个月累计位移预测", fontsize=10)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2, ls=":")
    save(fig, "F15_forecast_curves.png")

    # ---------------- F16 静态因子重要性 ----------------
    si = pd.read_csv(os.path.join(OUT, "tables", "static_importance.csv"))
    si = si.sort_values("mean_abs_delta_mm")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(si.feature, si.mean_abs_delta_mm, color=C_MAIN)
    ax.set_xlabel("置换后 |Δ预测| (mm)"); ax.set_title("静态环境因子重要性（置换法）", fontsize=10)
    ax.grid(axis="x", alpha=0.2, ls=":")
    save(fig, "F16_static_importance.png")


if __name__ == "__main__":
    main()
