# -*- coding: utf-8 -*-
"""Step2b: 用新补充的地质/水系/断层/高精度 DEM 数据，为每个 PS 点提取属性。

新增特征（对应申报书数据清单）：
  岩性类型     GLiM 全球岩性图（1:3.7M，中国地质科学院 1:50 万的开放替代）
  距河流距离   HydroRIVERS v1.0 亚洲水系（1:50 万全国水系矢量图的开放替代）
  距断层距离   GEM 全球活动断层数据库（1:25 万全国地质构造数据库的开放替代）
  高程/坡度    Copernicus DEM 30 m（ASTER GDEM 30 m 的开放替代，消除条带噪声）

输入: data/ps_points.parquet (lon, lat)
输出: data/ps_geo.parquet
"""
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

ROOT = r"E:\tianjindata"
SUP = os.path.join(ROOT, "data_supplement")
DATA = os.path.join(ROOT, "subsidence_project", "data")
UTM = "EPSG:32650"

LON0, LON1 = 116.88, 117.24
LAT0, LAT1 = 39.08, 39.32


def load_points():
    ps = pd.read_parquet(os.path.join(DATA, "ps_points.parquet"))
    if "lon" not in ps.columns:
        ps = ps.rename(columns={"longitude": "lon", "latitude": "lat"})
    gdf = gpd.GeoDataFrame(
        ps, geometry=gpd.points_from_xy(ps.lon, ps.lat), crs="EPSG:4326"
    )
    return gdf


def nearest_distance(points_utm, lines_utm):
    """点到线要素集合的最近距离（m）。"""
    if len(lines_utm) == 0:
        return np.full(len(points_utm), np.nan)
    lines_utm = lines_utm[lines_utm.geometry.notna()]
    rp = points_utm.geometry.apply(lambda g: g)  # 保留索引
    tree = lines_utm.sindex
    out = np.full(len(points_utm), np.nan)
    for i, (idx, pt) in enumerate(zip(points_utm.index, points_utm.geometry)):
        cand = list(tree.nearest(pt, 5))
        if not cand:
            continue
        d = [pt.distance(lines_utm.geometry.values[j]) for j in cand]
        out[i] = min(d)
    return out


def main():
    ps = load_points()
    n = len(ps)
    print(f"[pts] {n} 个 PS 点")

    # ---------- Copernicus DEM 30 m ----------
    dems = []
    for t in ["N39_00_E116_00", "N39_00_E117_00"]:
        p = os.path.join(SUP, "dem_cop30",
                         f"Copernicus_DSM_COG_10_{t}_DEM.tif")
        if os.path.exists(p):
            dems.append(p)
    if len(dems) == 2:  # 两幅相邻瓦片，按 bbox 读取
        from rasterio.merge import merge
        with rasterio.open(dems[0]) as a, rasterio.open(dems[1]) as b:
            arr, tr = merge([a, b])
            crs = a.crs
        dem = arr[0].astype(np.float32)
        dem[dem < -1000] = np.nan
        dem[dem == 0] = np.nan  # Copernicus 海洋/无数据为 0
        gy, gx = np.gradient(dem, 30.0, 30.0)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        rows, cols = rasterio.transform.rowcol(tr, ps.lon.values, ps.lat.values)
        rows = np.clip(rows, 0, dem.shape[0] - 1)
        cols = np.clip(cols, 0, dem.shape[1] - 1)
        ps["cop_elev"] = dem[rows, cols]
        ps["cop_slope"] = slope[rows, cols]
        print(f"[dem] Copernicus DEM {dem.shape} 采样完成")
    else:
        print("[dem] 缺少 Copernicus DEM，跳过")

    # ---------- HydroRIVERS ----------
    hr_dir = os.path.join(SUP, "hydrology", "HydroRIVERS_v10_as_shp")
    shps = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if f.endswith(".shp")] if os.path.isdir(hr_dir) else []
    if shps:
        riv = gpd.read_file(shps[0], bbox=(LON0, LAT0, LON1, LAT1))
        # 只保留 ≥3 级河流，避免人工渠系噪声
        if "ORD_FLOW" in riv.columns:
            riv = riv[riv.ORD_FLOW <= 6]
        riv_utm = riv.to_crs(UTM)
        txt = ps.to_crs(UTM)
        ps["dist_river_km"] = nearest_distance(txt, riv_utm) / 1000.0
        print(f"[river] HydroRIVERS 裁切后 {len(riv)} 条，距离完成")
    else:
        print("[river] 缺少 HydroRIVERS，跳过")

    # ---------- GEM 活动断层 ----------
    gpk = os.path.join(SUP, "geology", "gem_active_faults.gpkg")
    if os.path.exists(gpk):
        flt = gpd.read_file(gpk, bbox=(LON0, LAT0, LON1, LAT1))
        if len(flt):
            ps["dist_fault_km"] = nearest_distance(ps.to_crs(UTM), flt.to_crs(UTM)) / 1000.0
            print(f"[fault] 区内 GEM 断层 {len(flt)} 条，距离完成")
        else:
            print("[fault] 区内无 GEM 活动断层记录")
    else:
        print("[fault] 缺少 GEM 断层数据，跳过")

    # ---------- GLiM 岩性 ----------
    gdb = os.path.join(SUP, "geology", "LiMW_GIS2015")
    if os.path.isdir(gdb):
        lito_paths = [os.path.join(gdb, f) for f in os.listdir(gdb) if f.endswith(".gdb")]
        if lito_paths:
            layers = gpd.list_layers(lito_paths[0]).name.tolist()
            lito = None
            for ly in layers:
                try:
                    g = gpd.read_file(lito_paths[0], layer=ly,
                                      bbox=(LON0, LAT0, LON1, LAT1))
                except Exception:
                    continue
                if len(g) and any(g.geom_type.str.contains("Polygon")):
                    lito = g
                    break
            if lito is not None:
                col = next((c for c in lito.columns
                            if c.lower() in ("xx", "litho", "lithology", "class")), None)
                j = gpd.sjoin(ps, lito[[col, "geometry"]] if col else lito[["geometry"]],
                              how="left", predicate="within")
                j = j[~j.index.duplicated()]
                ps["litho"] = (j[col].values if col else "unknown")
                print(f"[litho] GLiM 类别分布:\n{ps['litho'].value_counts().head(10)}")
            else:
                print("[litho] GLiM 图层未识别")
    else:
        print("[litho] 缺少 GLiM 数据，跳过")

    keep = [c for c in ps.columns if c != "geometry"]
    ps[keep].to_parquet(os.path.join(DATA, "ps_geo.parquet"), index=False)
    print(f"[save] data/ps_geo.parquet  ({len(keep)} 列)")
    print(ps[keep].describe(include="all").T.to_string()[:2000])


if __name__ == "__main__":
    main()
