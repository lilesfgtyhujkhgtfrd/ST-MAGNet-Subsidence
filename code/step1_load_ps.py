# -*- coding: utf-8 -*-
"""Step1: 读取 7 个 PS-InSAR 分区 DBF，合并为统一点数据集并落盘为 parquet。

产物:
  data/ps_points.parquet   每行一个 PS 点: lon, lat, vel, coh, elev, 38 期 LOS 位移(mm)
  data/meta.json           影像期次日期等元数据
"""
import json
import os
import struct
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\tianjindata"
OUT = os.path.join(ROOT, "subsidence_project", "data")
os.makedirs(OUT, exist_ok=True)


def parse_dbf(path: str):
    """纯 numpy/struct 解析 Shapefile DBF，返回字段字典。"""
    with open(path, "rb") as f:
        raw = f.read()
    nrec = struct.unpack("<I", raw[4:8])[0]
    hlen = struct.unpack("<H", raw[8:10])[0]
    rlen = struct.unpack("<H", raw[10:12])[0]
    nfields = (hlen - 33) // 32
    fields = []
    for i in range(nfields):
        off = 32 + i * 32
        name = raw[off:off + 11].split(b"\0")[0].decode("latin1").strip()
        ftype = raw[off + 11:off + 12].decode("latin1")
        flen = raw[off + 16]
        fields.append((name, ftype, flen))
    body = np.frombuffer(raw[hlen:hlen + nrec * rlen], dtype=np.uint8).reshape(nrec, rlen)
    out = {}
    pos = 1  # 首字节为删除标记
    for name, ftype, flen in fields:
        blob = body[:, pos:pos + flen].copy().tobytes()
        if ftype == "N":
            arr = np.frombuffer(blob, dtype=f"S{flen}")
            arr = np.char.strip(arr)
            arr = np.where(arr == b"", b"0", arr)
            out[name] = arr.astype(np.float64)
        elif ftype == "C":
            arr = np.frombuffer(blob, dtype=f"S{flen}")
            out[name] = arr.astype(str)
        pos += flen
    return out


def main():
    import glob

    files = sorted(glob.glob(os.path.join(ROOT, "ps_insar", "*.dbf")))
    frames, dates_ref = [], None
    for f in files:
        d = parse_dbf(f)
        date_cols = sorted([k for k in d if k.startswith("D_")])
        if dates_ref is None:
            dates_ref = date_cols
        assert date_cols == dates_ref, f"日期列不一致: {os.path.basename(f)}"
        df = pd.DataFrame({
            "lon": d["lon"], "lat": d["lat"],
            "vel": d["velocity"], "coh": d["coherence"],
            "elev": d["Z"], "vp": d["Vprecision"],
            "tile": os.path.basename(f).replace(".dbf", ""),
        })
        for c in date_cols:
            df[c] = d[c]
        frames.append(df)
        print(f"[load] {os.path.basename(f)}  n={len(df):6d}  "
              f"lon {df.lon.min():.4f}~{df.lon.max():.4f}  lat {df.lat.min():.4f}~{df.lat.max():.4f}")

    ps = pd.concat(frames, ignore_index=True)
    print(f"[merge] 点数 = {len(ps)}")

    # 去重（分带重叠处同一点可能被重复导出）
    key = ps["lon"].round(6).astype(str) + "_" + ps["lat"].round(6).astype(str)
    dup = key.duplicated()
    print(f"[dedup] 重复点 = {int(dup.sum())}  ({dup.mean() * 100:.1f}%)")
    ps = ps.loc[~dup].reset_index(drop=True)
    print(f"[dedup] 去重后点数 = {len(ps)}")

    dates = [c.replace("D_", "") for c in dates_ref]
    epoch_cols = [f"e{i:02d}" for i in range(len(dates))]
    ps = ps.rename(columns={c: epoch_cols[i] for i, c in enumerate(dates_ref)})

    # 质量标记：速率饱和(±50 mm/a)为 SARscape 导出截断值，仅标记不删除
    ps["vel_clipped"] = (ps["vel"].abs() >= 49.99).astype(np.int8)
    meta = {
        "n_points": int(len(ps)),
        "n_epochs": len(dates),
        "dates": dates,
        "epoch_cols": epoch_cols,
        "tiles": sorted(ps["tile"].unique().tolist()),
        "vel_clipped_ratio": float(ps["vel_clipped"].mean()),
        "coh_min": float(ps["coh"].min()),
        "lon_range": [float(ps.lon.min()), float(ps.lon.max())],
        "lat_range": [float(ps.lat.min()), float(ps.lat.max())],
        "elev_range": [float(ps.elev.min()), float(ps.elev.max())],
        "note": "位移为 LOS 向 mm；速率单位 mm/a（SARscape 导出，绝对值≥50 被截断）",
    }
    ps.to_parquet(os.path.join(OUT, "ps_points.parquet"), index=False)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[save] {os.path.join(OUT, 'ps_points.parquet')}")
    print(f"[meta] 期次 {dates[0]} ~ {dates[-1]} ({len(dates)} 期); 截断速率占比 {meta['vel_clipped_ratio']*100:.1f}%")
    print(ps[["vel", "coh", "elev"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
