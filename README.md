 广域城市地下水超采区时空沉降趋势预测与驱动机制分析
本仓库为大创项目的全部可复现代码、数据中间产物与成果图件。核心工作是：以 PS-InSAR
时序形变场为基础，构建"水文地质单元 + 空间邻近/环境相似图结构"，用三分支图-注意力网络
（ST-MAGNet）预测未来 6/12 个月沉降，并通过门控权重、注意力、邻居贡献度与驱动因子置换
重要性解释沉降的驱动机制

## 1. 目录结构

```
subsidence_project/
├─ code/
│  ├─ step1_load_ps.py      # 解析 7 个 PS-InSAR 分区的 DBF，合并去重 → 669,373 个 PS 点
│  ├─ step2_env_features.py # DEM/土地利用/OSM 矢量 → 高程/坡度/距离/密度/地类等环境因子
│  ├─ step3_drivers.py      # NASA POWER 气象 + 水资源公报年际统计 → 月度驱动序列
│  ├─ step4_build_units.py  # 600 m 网格聚合水文地质单元 + 垂直向换算 + KMeans 地理亚区
│  ├─ step5_build_graph.py  # KNN 空间邻近 + 环境相似度 → 加权图
│  ├─ dataset.py            # 滑窗样本、标准化、两种训练/测试划分协议
│  ├─ model.py              # ST-MAGNet 与 8 种基线/消融模型
│  ├─ train_eval.py         # 训练、评估、外推预测（产出 results_all.csv / forecast.npz）
│  ├─ trend_baseline.py     # 线性趋势外推基线（写入 results_all.csv）
│  ├─ extra_runs.py         # 去时序消融的公平重跑 + ST-MAGNet 多种子稳定性实验
│  ├─ interpret.py          # 可解释性：门控、注意力、邻居遮挡、置换重要性、滞后相关
│  └─ figures.py            # 16 张成果图件
├─ data/                    # 中间数据（parquet/npz/csv/json，见 §3）
├─ outputs/
│  ├─ figures/              # 成果图件 PNG
│  ├─ tables/               # 可解释性结果表
│  ├─ results_all.csv       # 全部模型 × 两种协议的指标
│  ├─ forecast.npz          # 未来 12 个月逐月增量预测
│  └─ stmagn_final.pt       # 最终模型权重
└─ README.md
```

## 2. 运行环境与复现

```
Python 3.13 / numpy pandas scipy scikit-learn pyarrow
geopandas pyogrio rasterio pyproj
torch 2.15.0+cpu
matplotlib（中文字体 Microsoft YaHei）
```

按顺序执行（Windows PowerShell，建议 4 线程）：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:OMP_NUM_THREADS=4
cd e:\tianjindata\subsidence_project\code
python step1_load_ps.py      # ~2 min
python step2_env_features.py # ~3 min
python step3_drivers.py      # <1 min
python step4_build_units.py  # <1 min
python step5_build_graph.py  # <1 min
python train_eval.py         # 9 模型 × 2 协议 + 最终外推
python trend_baseline.py      # 线性趋势外推基线
python extra_runs.py          # 消融公平重跑 + 多种子稳定性
python interpret.py
python figures.py
```

