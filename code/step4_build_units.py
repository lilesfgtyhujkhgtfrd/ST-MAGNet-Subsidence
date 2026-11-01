# -*- coding: utf-8 -*-
"""Step4: 由 PS 点聚合水文地质单元，构建单元尺度时空样本。

- 单元: 研究区内规则网格（600 m），要求单元内 PS 点 ≥ 30 个
- 单元位移序列: 各期 LOS 位移中位数 -> 垂直向位移（按 38.5° 入射角换算）
- 静态环境属性: 高程/坡度/土地利用构成/距离类/建筑密度/相干性
- 动态驱动: 区域气象与地下水位/开采序列（step3 产物）

产物: data/units.npz, data/units.csv, data/samples.npz
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA = r"E:\tianjindata\subsidence_project\data"
CELL = 600.0          # m
MIN_PTS = 30
INC_DEG = 38.5        # Sentinel-1 IW 入射角（升轨）
K_ZONES = 4           # 地理亚区数


def main():
    ps = pd.read_parquet(os.path.join(DATA, "ps_env.parquet"))
    with open(os.path.join(DATA, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    drivers = pd.read_csv(os.path.join(DATA, "drivers.csv"), dtype={"idx": str})
    drivers["ym"] = drivers.iloc[:, 0].astype(str)
    drivers = drivers.set_index("ym")

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    x, y = tr.transform(ps.lon.values, ps.lat.values)

    # ---- 网格单元 ----
    x0, y0 = x.min(), y.min()
    ci = ((x - x0) // CELL).astype(np.int32)
    cj = ((y - y0) // CELL).astype(np.int32)
    ncols = cj.max() + 1
    cid = cj * ncols + ci
    ps["cid"] = cid
    ps["x"], ps["y"] = x, y

    grp = ps.groupby("cid")
    counts = grp.size()
    keep = counts[counts >= MIN_PTS].index
    print(f"[unit] 网格 {int(ci.max())+1}x{int(cj.max())+1}, 有PS单元 {len(counts)}, "
          f"≥{MIN_PTS}点单元 {len(keep)}")

    epochs = meta["epoch_cols"]
    cos_inc = np.cos(np.deg2rad(INC_DEG))
    recs = []
    for c in keep:
        g = ps[ps.cid == c]
        disp = np.nanmedian(g[epochs].values, axis=0) / cos_inc  # LOS -> 垂直向 mm
        rec = {
            "cid": int(c),
            "n_pts": int(len(g)),
            "x": float(g.x.mean()), "y": float(g.y.mean()),
            "lon": float(g.lon.mean()), "lat": float(g.lat.mean()),
            "elev": float(g.dem_elev.mean()), "slope": float(g.slope.mean()),
            "coh": float(g.coh.mean()), "vel": float(g.vel.mean()) / cos_inc,
            "dist_road": float(g.dist_road.mean()), "dist_major_road": float(g.dist_major_road.mean()),
            "dist_water": float(g.dist_water.mean()), "dist_rail": float(g.dist_rail.mean()),
            "dist_center": float(g.dist_center.mean()),
            "building_density": float(g.building_density.mean()),
            "lu_urban": float((g.landuse == 8).mean()),
            "lu_crop": float((g.landuse == 1).mean()),
            "lu_water": float((g.landuse == 5).mean()),
            "lu_other": float((~g.landuse.isin([1, 5, 8])).mean()),
        }
        for i, e in enumerate(epochs):
            rec[e] = float(disp[i])
        recs.append(rec)
    units = pd.DataFrame(recs).sort_values("cid").reset_index(drop=True)

    # 位移序列去跳变：与前后均值偏差>3倍MAD的观测用插值替换
    D = units[epochs].values.astype(np.float64)
    diff = np.diff(D, axis=1)
    med = np.median(diff, axis=1, keepdims=True)
    mad = np.median(np.abs(diff - med), axis=1, keepdims=True) + 1e-3
    jump = np.abs(diff - med) > 6 * mad
    for i, j in zip(*np.where(jump)):
        k = j + 1
        left = D[i, k - 1]
        right = D[i, k + 1] if k + 1 < D.shape[1] else left
        D[i, k] = 0.5 * (left + right)
    print(f"[clean] 修正跳变观测 {int(jump.sum())} / {jump.size} ({jump.mean()*100:.2f}%)")
    units[epochs] = D

    # ---- 地理亚区（KMeans：位置 + 城市化强度） ----
    Z = np.c_[units.x / 1000, units.y / 1000, units.building_density * 5, units.dist_center]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    km = KMeans(n_clusters=K_ZONES, n_init=20, random_state=42).fit(Z)
    units["zone"] = km.labels_
    # 亚区命名：按质心纬度自北向南排序
    zone_stats = units.groupby("zone").agg(lat=("lat", "mean"), urb=("building_density", "mean"))
    order = zone_stats.sort_values("lat", ascending=False).index.tolist()
    names = ["北亚区", "中北亚区", "中南亚区", "南亚区"][:len(order)]
    mapping = {z: names[i] if i < len(names) else f"亚区{i}" for i, z in enumerate(order)}
    units["zone_name"] = units["zone"].map(mapping)
    print("[zone]", units.groupby("zone_name").size().to_dict())

    # ---- 驱动序列对齐 ----
    years = [int(s[:4]) for s in meta["dates"]]
    months = [int(s[4:6]) for s in meta["dates"]]
    ym = [f"{y}{m:02d}" for y, m in zip(years, months)]
    drv = drivers.reindex(pd.Index(ym))
    # 个别缺测月份前向填充
    drv = drv.ffill().bfill()
    drv_cols = ["precip_mm_z", "temp_c_z", "gw_level_anom_m_z", "gw_abstraction_yi_m3_z", "soil_wet_z"]
    Drv = drv[drv_cols].values.astype(np.float32)

    units.to_csv(os.path.join(DATA, "units.csv"), index=False, encoding="utf-8")
    np.savez(os.path.join(DATA, "units.npz"),
             D=D.astype(np.float32), Drv=Drv.astype(np.float32),
             cid=units.cid.values, x=units.x.values, y=units.y.values,
             zone=units.zone.values, epochs=np.array(epochs), ym=np.array(ym),
             driver_cols=np.array(drv_cols))
    with open(os.path.join(DATA, "units_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cell_m": CELL, "n_units": int(len(units)), "min_pts": MIN_PTS,
                   "incidence_deg": INC_DEG, "n_epochs": len(epochs),
                   "epochs": meta["dates"], "ym": ym, "driver_cols": drv_cols,
                   "K_ZONES": K_ZONES, "zone_names": sorted(set(mapping.values()))},
                  f, ensure_ascii=False, indent=2)

    print(units[["vel", "coh", "n_pts", "elev", "building_density"]].describe().round(2).to_string())
    print("[save] data/units.csv / units.npz")


if __name__ == "__main__":
    main()
