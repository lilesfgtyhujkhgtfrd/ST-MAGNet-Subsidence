# -*- coding: utf-8 -*-
"""Step2: 为每个 PS 点提取静态环境特征。

数据源:
  DEM     -> 高程、坡度 (ASTGTM 30m)
  CLCD    -> 土地利用类型 (30m, 2015)
  OSM     -> 距道路/铁路/水系距离、建筑密度

产物: data/ps_env.parquet
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

ROOT = r"E:\tianjindata"
DATA = os.path.join(ROOT, "subsidence_project", "data")
RAW = os.path.join(DATA, "raw")
UTM = 32650  # WGS84 / UTM zone 50N

# 研究区栅格范围（含 2 km 缓冲）
LON0, LON1 = 116.89, 117.22
LAT0, LAT1 = 39.09, 39.30
RES = 25.0  # m


def to_utm(lon, lat):
    """经纬度 -> UTM50N（pyproj 精确投影）。"""
    from pyproj import Transformer
    global _TR
    if "_TR" not in globals():
        _TR = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM}", always_xy=True)
    return _TR.transform(lon, lat)


def sample_raster(src, xs, ys):
    """按 UTM 坐标采样（最近邻，返回数组）。"""
    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, src.height - 1)
    cols = np.clip(np.asarray(cols), 0, src.width - 1)
    return src.read(1)[rows, cols]


def build_grid():
    x0, y0 = to_utm(LON0, LAT1)   # 左上角
    x1, y1 = to_utm(LON1, LAT0)   # 右下角
    width = int(np.ceil((x1 - x0) / RES))
    height = int(np.ceil((y0 - y1) / RES))
    transform = from_origin(x0, y0, RES, RES)
    return transform, width, height, (x0, y1, x1, y0)


def main():
    ps = pd.read_parquet(os.path.join(DATA, "ps_points.parquet"))
    with open(os.path.join(DATA, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    x, y = to_utm(ps.lon.values, ps.lat.values)
    ps["x_utm"], ps["y_utm"] = x, y

    transform, W, H, bounds = build_grid()
    print(f"[grid] {W}x{H} @ {RES}m, bounds={bounds}")

    # ---------- DEM ----------
    dem = None
    for fn in ["ASTGTM_N39E117Y.img", "ASTGTM_N38E117J.img"]:
        p = os.path.join(RAW, fn)
        with rasterio.open(p) as src:
            print(f"[dem] {fn}: {src.width}x{src.height} res={src.res} crs={src.crs}")
            if dem is None:
                dem = src.read(1).astype(np.float32)
                dem_transform = src.transform
                dem_nodata = src.nodata
            else:
                pass
    # 研究区全部在 N39E117（39~40N），仅用该幅
    nod = dem_nodata if dem_nodata is not None else -9999
    dem[dem == nod] = np.nan
    dem[dem < -1000] = np.nan
    # 坡度（30m 网格）
    gy, gx = np.gradient(dem, 30.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    d_rows, d_cols = rasterio.transform.rowcol(dem_transform, x, y)
    d_rows = np.clip(d_rows, 0, dem.shape[0] - 1)
    d_cols = np.clip(d_cols, 0, dem.shape[1] - 1)
    ps["dem_elev"] = dem[d_rows, d_cols]
    ps["slope"] = slope[d_rows, d_cols]

    # ---------- CLCD 土地利用（Albers -> UTM，重投影至研究区网格） ----------
    clcd_p = os.path.join(ROOT, "CLCD_v01_2015_albert_tianjin.tif")
    with rasterio.open(clcd_p) as src:
        print(f"[clcd] {src.width}x{src.height} res={src.res} crs={src.crs.to_string()[:40]}...")
        dst = np.zeros((H, W), dtype=np.uint8)
        from rasterio.warp import reproject, Resampling
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=f"EPSG:{UTM}",
            resampling=Resampling.nearest,
        )
    rr, cc = rasterio.transform.rowcol(transform, x, y)
    rr = np.clip(rr, 0, H - 1); cc = np.clip(cc, 0, W - 1)
    ps["landuse"] = dst[rr, cc].astype(np.int16)
    print("[clcd] 类别分布:", dict(zip(*[v.tolist() for v in np.unique(ps.landuse, return_counts=True)])))

    # ---------- OSM 矢量 -> 距离栅格 ----------
    import geopandas as gpd
    gpkg = os.path.join(ROOT, "tianjin-261029-free.gpkg", "tianjin.gpkg")

    def geoms(layer, bbox_buf=0.03):
        g = gpd.read_file(gpkg, layer=layer,
                          bbox=(LON0 - bbox_buf, LAT0 - bbox_buf, LON1 + bbox_buf, LAT1 + bbox_buf))
        return g.to_crs(UTM)

    def dist_raster(gdf):
        if len(gdf) == 0:
            return np.full((H, W), 1e5, np.float32)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and geom.is_valid]
        mask = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype="uint8")
        return ndimage.distance_transform_edt(mask == 0, sampling=(RES, RES)).astype(np.float32)

    roads = geoms("gis_osm_roads_free")
    major = roads[roads.fclass.isin(["motorway", "trunk", "primary", "secondary", "motorway_link",
                                     "trunk_link", "primary_link", "secondary_link"])]
    rails = geoms("gis_osm_railways_free")
    wways = geoms("gis_osm_waterways_free")
    water = geoms("gis_osm_water_a_free")
    bld = geoms("gis_osm_buildings_a_free")
    print(f"[osm] roads={len(roads)} major={len(major)} rail={len(rails)} "
          f"waterway={len(wways)} water={len(water)} buildings={len(bld)}")

    r_dist = dist_raster(roads)
    r_major = dist_raster(major)
    rail_dist = dist_raster(rails)
    water_dist = dist_raster(pd.concat([wways[["geometry"]], water[["geometry"]]]))
    # 建筑密度：500m 窗口内建筑面积占比（栅格化面积 -> 高斯平滑）
    bpolys = [(g, 1) for g in bld.geometry if g is not None and g.is_valid]
    bmask = rasterize(bpolys, out_shape=(H, W), transform=transform, fill=0, dtype="float32")
    sigma_px = 500.0 / RES / 2.5
    bdens = ndimage.gaussian_filter(bmask, sigma_px, mode="constant")
    bdens = bdens / (bdens.max() + 1e-6)

    def grid_sample(r):
        rr, cc = rasterio.transform.rowcol(transform, x, y)
        rr = np.clip(rr, 0, H - 1); cc = np.clip(cc, 0, W - 1)
        return r[rr, cc]

    ps["dist_road"] = grid_sample(r_dist)
    ps["dist_major_road"] = grid_sample(r_major)
    ps["dist_rail"] = grid_sample(rail_dist)
    ps["dist_water"] = grid_sample(water_dist)
    ps["building_density"] = grid_sample(bdens)
    cx, cy = to_utm(117.1904, 39.1251)
    ps["dist_center"] = np.hypot(ps.x_utm - cx, ps.y_utm - cy) / 1000.0  # km

    ps.drop(columns=["x_utm", "y_utm"]).to_parquet(os.path.join(DATA, "ps_env.parquet"), index=False)
    print(ps[["dem_elev", "slope", "landuse", "dist_road", "dist_water",
              "dist_rail", "building_density", "dist_center"]].describe().round(2).to_string())
    print("[save] data/ps_env.parquet")


if __name__ == "__main__":
    main()
