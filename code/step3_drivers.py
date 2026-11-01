# -*- coding: utf-8 -*-
"""Step3: 构建动态诱发因子月度序列。

真实在线数据:
  NASA POWER 月度气象 (PRECTOTCORR mm/day, T2M ℃, RH2M %, GWETROOT) —— 4 个格点区域平均
文献/公报统计重建:
  深层地下水年开采量、深层地下水位年际变化（2017-2020，来源见 data/driver_sources.json）
  月度分配采用华北平原典型季节形态（采暖季开采强度高、春季水位回升快）

产物: data/drivers.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd

DATA = r"E:\tianjindata\subsidence_project\data"

# 发布的年际统计（来源: 天津市水资源公报 2018/2019/2020；天津市水务局 2025 新闻；
# FMAC 2023 论文报道）。deep_abstraction 单位 亿 m3；gw_level_anom 单位 m，以 2017 年初为 0
DRIVER_SOURCES = {
    "deep_abstraction_yi_m3": {
        "2017": 1.80, "2018": 1.50, "2019": 1.13, "2020": 0.65,
        "note": "2018=1.5(公报/水务局口径),2019=1.13,2020=0.65(2020公报,同比-0.48);2017 由趋势估计",
    },
    "gw_level_anom_m": {
        "2017": -0.50, "2018": 1.10, "2019": 3.50, "2020": 6.12,
        "note": "2020 年深层水位同比+2.62m(2020公报);2018-2024 连续6年回升累计11.8m(水务局2025);"
                "第三承压含水层自2018年以约2.1m/a回升(FMAC,2023)",
    },
    "annual_subsidence_mm": {
        "2017": 20.0, "2018": 18.0, "2019": 12.0, "2020": 8.0,
        "note": "平原区平均地面沉降量由20mm降至1mm(2018-2024,水务局2025);年际值为区间内插",
    },
}


def load_power():
    rows = {}
    for f in glob.glob(os.path.join(DATA, "raw", "power", "*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        par = d["properties"]["parameter"]
        for key, val in par.items():
            s = pd.Series({k: v for k, v in val.items()})
            rows.setdefault(key, []).append(s)
    out = {}
    for k, lst in rows.items():
        df = pd.concat(lst, axis=1).mean(axis=1)
        out[k] = df
    df = pd.DataFrame(out)
    df.index.name = "yyyymm"
    # POWER 的年度汇总行月份记为 13，剔除
    df = df[[s.isdigit() and 1 <= int(s[4:]) <= 12 for s in df.index]]
    return df


def main():
    power = load_power()
    df = pd.DataFrame(index=sorted(power.index))
    # 月总量/月均值
    df["precip_mm"] = (power["PRECTOTCORR"] * power.index.map(
        lambda s: pd.Period(f"{s[:4]}-{s[4:]}").days_in_month) ).round(1)
    df["temp_c"] = power["T2M"].round(2)
    df["rh_pct"] = power["RH2M"].round(2)
    df["soil_wet"] = power["GWETROOT"].round(3)
    df["month"] = [int(s[4:]) for s in df.index]
    df["year"] = [int(s[:4]) for s in df.index]

    # ---- 统计重建: 年际量 -> 月度 ----
    years = [int(y) for y in DRIVER_SOURCES["deep_abstraction_yi_m3"] if y.isdigit()]
    abst_annual = {int(k): v for k, v in DRIVER_SOURCES["deep_abstraction_yi_m3"].items()
                   if k.isdigit()}
    lvl_annual = {int(k): v for k, v in DRIVER_SOURCES["gw_level_anom_m"].items() if k.isdigit()}

    # 开采量月度分配: 采暖季(11-3月)权重高
    w = np.array([1.35, 1.30, 1.20, 0.90, 0.85, 0.80, 0.85, 0.90, 0.90, 0.85, 1.20, 1.35])
    w = w / w.sum()
    abst_m = []
    for ym in df.index:
        y = int(ym[:4])
        if y in abst_annual:
            abst_m.append(abst_annual[y] * w[int(ym[4:]) - 1])
        else:
            abst_m.append(np.nan)
    df["gw_abstraction_yi_m3"] = abst_m

    # 水位: 年际值线性内插到月，叠加季节波动（3月最高、8月最低）
    lvl_y = []
    for y in years:
        lvl_y.append(lvl_annual[y])
    # 以该年 7 月为锚点（index 0 = 2016-01）
    anchors = [(y - 2016) * 12 + 6 for y in years]
    all_idx = np.arange(len(df))
    lvl = np.interp(all_idx, anchors, lvl_y, left=lvl_y[0] - 0.02 * (anchors[0] - all_idx[0])
                    if all_idx[0] < anchors[0] else None)
    # 两端线性外推
    lvl = np.where(all_idx < anchors[0],
                   lvl_y[0] + (lvl_y[1] - lvl_y[0]) / 12 * (all_idx - anchors[0]), lvl)
    lvl = np.where(all_idx > anchors[-1],
                   lvl_y[-1] + (lvl_y[-1] - lvl_y[-2]) / 12 * (all_idx - anchors[-1]), lvl)
    # 季节项在 7 月归零，保证 7 月值等于年际锚点
    ph = 2 * np.pi * (df["month"].values - 3) / 12 + np.pi / 2
    ph_jul = 2 * np.pi * (7 - 3) / 12 + np.pi / 2
    seasonal = 0.55 * (np.sin(ph) - np.sin(ph_jul))
    df["gw_level_anom_m"] = (lvl + seasonal).round(3)
    df["gw_level_depth_m"] = (32.0 - df["gw_level_anom_m"]).round(3)  # 视深层水位基准埋深32m

    # 标准化驱动量（模型输入用）
    for c in ["precip_mm", "temp_c", "gw_level_anom_m", "gw_abstraction_yi_m3", "soil_wet"]:
        v = df[c].astype(float)
        df[c + "_z"] = ((v - v.mean()) / (v.std() + 1e-9)).round(4)

    df.to_csv(os.path.join(DATA, "drivers.csv"), encoding="utf-8")
    with open(os.path.join(DATA, "driver_sources.json"), "w", encoding="utf-8") as f:
        json.dump(DRIVER_SOURCES, f, ensure_ascii=False, indent=2)

    print(df.loc["201703":"202004", ["precip_mm", "temp_c", "gw_level_anom_m",
                                     "gw_abstraction_yi_m3"]].to_string())
    print("[save] data/drivers.csv")


if __name__ == "__main__":
    main()
