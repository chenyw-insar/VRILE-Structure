# VRILE Stage 3：Pan-Arctic / Local Contribution Attribution 实验方案

## 0. 定位、环境与当前状态

顶层流程：

```text
Stage 1  raw basis
Stage 2  major_severe / mechanism / evidence
Stage 3  pan-Arctic / local contribution attribution
StageX   GMT geospatial rendering；Stage 1/2/3 后可重跑
```

测试阶段：

```bash
conda activate test
```

随后使用 ambient `python`。当前不使用 `PY_ENV` / `conda run`；发行版再统一环境合同。

锁定语义：

```text
ΔSIE(date) = SIE(date+1) - SIE(date-2)
_5p = lower 5th percentile
primary pan-Arctic anchor = Stage 1 unique event date
unique event date = adjacent event block final row
pan-Arctic CDR spatial window = 5-day ending on event date
local detector windows = 3/5/7/10-day
NSIDC-0780 code 0 = non_region_ocean
```

Batch 0–1R2 已通过：

```text
G02135 required GeoTIFF = 196/196

D1a closure:
coverage = 99/99
median relative difference = 0.000923
P90 relative difference = 0.001660
sign agreement = 1.000
Spearman rho = 0.999913
gate = PASS

CDR concordance:
coverage = 99/99
median relative difference = 0.142734
P90 relative difference = 0.364805
sign agreement = 1.000
Spearman rho = 0.733144

route = exact_detector_attribution
```

2026-07-08 初始 Batch 2 overlap layer 曾存在 lifetime-footprint temporal-alignment bug。Batch 2R 已完成修复并通过代码/输出一致性审核。

Batch 2R 正式结果：

```text
primary Track S alignment =
patch interval intersects [T-5,T]
AND patch.date <= T

events affected by repair = 99/99
future patches excluded = 5165
union-footprint cell reduction:
median = 0.246666
P90    = 0.332693
max    = 0.482708

pan×local rows = 2806
major_severe rows = 382
component pairs = 4464

strict footprint reconciliation = 35559/35559 PASS
unique-event membership reconciliation = PASS

Track D internal budget = 99/99 PASS
Track D vs D1a reconstruction = 99/99 PASS
source-status activity fraction median/P90/max = 0/0/0

Track S fields = 99/99
Track S SIC budget closure = 99/99 PASS
CDR loss components = 3463
component_resolved_sic_loss_fraction:
median = 0.837049
P10    = 0.788244
P90    = 0.897593
```

因此：

```text
Batch 2R = PASS
```

2026-07-09 Batch 3 经 Batch 3R / 3R2 / 3R3 bounded acceptance repair 与独立 code/output audit 后正式通过并冻结。

Batch 3R3 最终验收：

```text
strict component reconciliation = 3463/3463 PASS
event resolved-loss closure = 99/99 PASS

detector persisted partition reconstruction = 99/99 PASS
CDR persisted partition reconstruction = 99/99 PASS

detector class-code regression = 99/99 PASS
CDR class-code regression = 99/99 PASS
Track D contribution regression = 99/99 PASS
Track S contribution regression = 99/99 PASS

CDR required files = 197/197 readable
unique full-grid signature = 1
unique CF CRS signature = 1
unique WKT CRS signature = 1
CF documented CRS contract = 197/197 PASS
WKT documented CRS contract = 197/197 PASS

CF/WKT actual-grid geolocation delta:
median = 0 m
P99    = 1.7700836033345725e-9 m
max    = 2.8318576212474325e-9 m

full Stage 3 run = PASS
log = logs/stage3_panarctic_local_contribution_20260709_154429.log
```

`pyproj.CRS.equals()` 对 CF/WKT representation仍返回 false，但两侧均满足 G02202 northern-grid documented projection contract，且在实际 448×304 grid上的差异仅为 numerical-representation level；不再作为 Stage 3 blocker。

当前状态：

```text
Batch 2R = PASS
Batch 3R3 = PASS
Batch 3 outputs = frozen
Stage 3 HOLD = lifted
current work = Batch 4A matched controls / co-loss / spatial compensation
```

Batch 4 不得修改已冻结的 Batch 3 class partitions、contribution budgets、region budgets或spatial-composition metrics。

另锁定一个 Track-specific alignment 边界：

> Batch 2R 的 primary local footprint 是 Track S 的 `[T-5,T]` no-look-ahead footprint。其 `detector_*_in_local_footprint` 字段只表示“Track S-aligned footprint 与 Track D field 的交叠诊断”，不得直接作为 Batch 3 exact detector-aligned class attribution input。

Batch 3 必须分别建立：

```text
M_local_detector(T)
M_local_cdr(T)
```

不得复用同一 temporal footprint完成两个 Track 的 class partition。

`exact_detector_attribution` 只适用于 G02135 detector-aligned extent decomposition。G02202/CDR 是独立的 5-day continuous SIC spatial-composition track。

### 0.1 Route 决策矩阵

| route | Track D | Track S | 后续 |
|---|---|---|---|
| `exact_detector_attribution` | 运行；extent budget closure 为硬 gate | 运行 | 允许进入当前 Batch 2R 及后续 |
| `cdr_based_spatial_decomposition` | 不作为 exact attribution 来源 | 运行 | 可走独立 degraded CDR-only 路线；禁止 Track D/exact wording |
| `detector_source_diagnose_first` | 不运行归因 | 不运行归因 | 停止，先诊断 detector source |

**当前 Batch 3 只针对已验证的 `exact_detector_attribution` route。** 若 route 改变，当前 shell 必须停止并报告；不得半实现 `cdr_based_spatial_decomposition` 分支.

## 1. 科学问题与总体假说

工作假说：

> **部分 pan-Arctic VRILE 是多个地理上分散的局地快速海冰损失经空间积分形成的 aggregate extreme；单一代表位置会丢失其多中心组成与空间补偿结构。**

Stage 3 回答：

1. pan-Arctic VRILE 是否经常具有 multi-center loss structure？
2. 哪些局地 rapid-loss events 与 pan-Arctic loss components 在真实栅格上重叠？
3. `major_severe`、`severe non-major`、`broad non-severe` 和 residual 分别解释多少 gross loss？
4. gross loss 有多少被其他区域 gain 空间补偿？
5. 排除同期 `major_severe` footprint 后，剩余北极或其他区域是否仍存在异常 co-loss？
6. 强局地事件未形成 pan-Arctic VRILE 时，是局地贡献不足、缺乏同步 co-loss，还是空间 gain compensation？
7. 在数据允许时，局地 SIC 变化有多少与海冰运动输运/发散一致？

Stage 3 是 spatial composition / budget / kinematic diagnostic，不作确定因果归因。

---

## 2. 三类时间语义严格分离

### 2.1 Detector metric

```text
ΔSIE(date) = SIE(date+1) - SIE(date-2)
```

数据：

```text
G02135 / Sea Ice Index
```

用途：

```text
pan-Arctic VRILE detection
exact detector-aligned extent decomposition
```

### 2.2 Pan-Arctic CDR spatial field

```text
start = unique event date - 5 days
end   = unique event date
```

数据：

```text
G02202 / CDR SIC
```

用途：

```text
continuous SIC loss/gain
loss components
multi-center structure
local-event raster overlap
```

### 2.3 Local patch detector

```text
3 / 5 / 7 / 10-day SIC-loss windows
```

因此：

```text
3-day detector metric
≠ 5-day pan-Arctic CDR spatial field
≠ local multi-window patch detection
```

不得混用。

Primary pan-Arctic anchor：

```text
outputs/reproduce_sie/vrile_events_unique_both_jja_5p.csv::date
```

Strongest-member date 仅作 sensitivity。

### 2.4 Track-specific local-footprint alignment

Batch 3 必须维护两套独立 temporal alignment。

#### Track D detector alignment

Detector field：

```text
[T-2, T+1]
```

Local patch进入 Track D footprint当且仅当：

```text
patch.start_date <= T+1
patch.date       >= T-2
patch.date       <= T+1
```

即 patch interval 与 detector window相交，且 patch end不超过 detector window end。

记为：

```text
M_local_detector(T)
```

注意：Stage 1 canonical ULE来自 5-day local patches。因此这里的“exact”只指 **G02135 detector field reconstruction/exact target budget**；local footprint仍是 detector-window-aligned spatial association，不作 causal attribution。

#### Track S CDR alignment

CDR field：

```text
[T-5, T]
```

Primary local patch：

```text
patch.start_date <= T
patch.date       >= T-5
patch.date       <= T
```

最后一条是 no-look-ahead gate。

记为：

```text
M_local_cdr(T)
```

严禁：

```text
use M_local_cdr for Track D class attribution
use M_local_detector for Track S class attribution
```

Batch 2R existing overlap tables继续作为 Track S-aligned relation/component diagnostics。

---

## 3. 数据合同

### 3.1 主数据

```text
data/raw/nsidc_sie/N_seaice_extent_daily_v4.0.csv
data/raw/nsidc_sic/<year>/*.nc
data/raw/nsidc_region_masks/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc

outputs/reproduce_sie/
outputs/local_vrile/
outputs/local_vrile_enhanced/
outputs/local_vrile_severe/
outputs/local_vrile_major_severe/
```

### 3.2 NSIDC-0771 cell area

正式文件：

```text
data/raw/nsidc_ancillary/
NSIDC0771_CellArea_PS_N25km_v1.1.nc
```

变量 `cell_area`，实际单位 `meters^2`，统一转 km²。

所有 extent / SIC-loss area integration 使用逐格点面积。禁止回退为固定 `625 km²`。

Grid compatibility：

```text
grid_index_equal
AND
(
  exact CRS match
  OR sampled geolocation delta <= min(1000 m, 0.05 × grid_step)
)
```

当前：

```text
shape/x/y equal
grid_index_equal = true
CRS exact unequal
sampled geolocation delta ≈ 134.863 m
threshold = 1000 m
compatible = true
```

`test` 环境若有 `pyproj`，优先用 `pyproj.CRS/Transformer`；否则保留 approximate fallback，并记录 `crs_delta_method`。

### 3.3 G02135 daily extent GeoTIFF

本地：

```text
data/raw/nsidc_g02135_geotiff/
└── north/daily/geotiff/
    └── <year>/<MM_Mon>/
        └── N_YYYYMMDD_extent_v*.tif
```

Downloader：

```text
data_download/download_stage3_required_data.py
```

当前：

```text
196/196 required detector dates
Failed=0
```

日期仅来自 99 个 unique events 的 `date-2` 与 `date+1`。当前不要求完整 JJA G02135 archive。

Manifest 合法状态：

```text
downloaded
skipped_valid
ok
```

### 3.4 D2 source grids

`NSIDC-0051 / NSIDC-0803` 仅作 D1a mismatch fallback。当前不下载、不实现。

---

## 4. 双 Track 框架

### 4.1 Track D：detector-aligned extent budget

数据：

```text
G02135 daily extent GeoTIFF
date-2 → date+1
```

方法合同：

```text
exact_attribution_scope =
g02135_detector_aligned_extent_only
```

用途：

> **定量描述局地 footprint 与 pan-Arctic detector-aligned extent loss 的空间对应。**

#### Extent-state accounting

G02135 classes：

```text
1   sea ice
0   ocean
253 coast
254 land
255 missing
```

Physical extent transition 仅定义于 start/end 均为 `{0,1}` 的格点：

```text
physical loss = 1 → 0
physical gain = 0 → 1
```

不得把 `255`、coast 或 land 当作 ocean。

定义 binary ice indicator：

```text
I = 1 if class == 1
I = 0 otherwise
```

对非 `{0,1}↔{0,1}`、且 `I_start != I_end` 的格点单独记录 source-status transition：

```text
detector_source_status_ice_loss_km2
detector_source_status_ice_gain_km2
detector_source_status_transition_net_km2
=
source_status_ice_gain - source_status_ice_loss
```

这样既不把 source-status transition 解释成物理 gain/loss，又保留 detector source 的完整 binary-ice accounting。

#### 两层 closure

**A. Field-budget internal closure：硬 gate**

由同一对 GeoTIFF 和同一 cell-area grid 计算：

```text
detector_gross_extent_gain_km2
-
detector_gross_extent_loss_km2
+
detector_source_status_transition_net_km2
≈
detector_geotiff_delta_sie_km2
```

其中：

```text
detector_geotiff_delta_sie_km2
=
(SIE_geotiff_end - SIE_geotiff_start) × 1,000,000
```

该式是同源逐像元 budget identity。使用：

```text
np.isclose(rtol=1e-12, atol=1e-6 km²)
```

99/99 events 必须 PASS。

**B. Detector-source fidelity：沿用 D1a gate**

```text
detector_geotiff_delta_sie_km2
vs
reported_raw_delta_sie_km2
```

其中：

```text
reported_raw_delta_sie_km2
=
raw_delta_sie × 1,000,000
```

该比较允许 Sea Ice Index CSV precision / source formatting 带来的小差异，继续使用 Batch 0–1R2 已通过的 D1a engineering gate；不得误写成 floating-point exact equality。

每个 event 至少输出：

```text
detector_gross_extent_loss_km2
detector_gross_extent_gain_km2
detector_source_status_ice_loss_km2
detector_source_status_ice_gain_km2
detector_source_status_transition_net_km2
detector_geotiff_delta_sie_km2
reported_raw_delta_sie_km2
detector_budget_closure_error_km2
detector_source_relative_difference
missing_touch_area_km2
other_class_touch_area_km2
```

正式术语统一为：

```text
budget closure
extent budget
contribution budget
```

不得称 `mass balance`，因为本阶段未估算 sea-ice mass/volume。

### 4.2 Track S：CDR 5-day SIC spatial composition

数据：

```text
G02202 / CDR SIC
event date - 5 days → event date
```

方法合同：

```text
cdr_spatial_composition_scope =
g02202_5day_sic
```

用途：

```text
continuous SIC loss/gain
loss components
multi-center structure
local footprint overlap
regional co-loss
```

不得称 exact G02135 contribution。

Batch 0–1R2 的 CDR concordance是 cross-product diagnostic：

```text
sign agreement = 1.000
Spearman rho = 0.733144
```

它说明损失方向一致，但事件幅度排序只有中等一致性；不是 blocker，也不要求继续“修”到 rho≈1。

2025 `n=3, rho=0.5` 仅保留 QA，不作 source-transition interpretation。

---

## 5. 代码架构原则

Stage 3 采用：

> **semantic step scripts + shared scientific package**

### 5.1 Semantic entry scripts

```text
scripts/stage3_panarctic_local_contribution.py
scripts/stage3_detector_extent_fields.py
scripts/stage3_local_footprints.py
scripts/stage3_cdr_loss_fields.py
scripts/stage3_raster_overlap.py

# Batch 3 current
scripts/stage3_batch3_preflight.py
scripts/stage3_contribution_budget.py
scripts/stage3_spatial_composition_metrics.py

# Later
scripts/stage3_coloss_controls.py
scripts/stage3_kinematic_budget.py
```

每个 script只负责：

```text
validate inputs
call shared scientific functions
write named outputs
emit step-level QA
```

### 5.2 Shared package

```text
src/vrile/local_objects.py

src/vrile/stage3/
├── __init__.py
├── grid.py
├── fields.py
├── footprints.py
├── overlap.py
├── budget.py
└── metrics.py
```

Batch 3 只创建实际使用的 `budget.py` / `metrics.py`；不提前创建 controls/kinematics空模块。

原则：

- Stage 1 / Stage 3 共用 detector primitive与 merge primitive；
- Track D / Track S alignment必须由 shared function显式区分；
- grid/mask/cell-area/class-partition逻辑不得在 wrappers间复制；
- foundation不继续扩大为 giant core；
- Shell只负责编排步骤和 gate。

---

## 6. Batch 2 前置 QA

Batch 2 开头直接完成，不单开 repair batch。

### 6.1 Test environment gate

```bash
conda activate test
```

Stage 3 shell 检查：

```text
CONDA_DEFAULT_ENV == test
```

否则退出。

日志：

```text
CondaEnv
Python
```

### 6.2 Manifest status

G02135 inventory 接受：

```text
downloaded
skipped_valid
ok
```

### 6.3 全 required GeoTIFF signature QA

扫描 196 个 required files：

```text
width
height
pixel_scale_x
pixel_scale_y
tiepoint_x
tiepoint_y
EPSG
grid signature
```

输出：

```text
g02135_geotiff_grid_signature_audit.csv
```

Gate：

```text
196/196 readable
single band
inventory valid
unique_grid_signature_count == 1
```

否则 hard fail。

---

## 7. Local footprint 与 membership 恢复

现有 local summary CSV 无 cell membership。禁止使用 centroid/buffer/region polygon 替代真实 footprint。

### 7.1 Shared primitives：已建立，Batch 2R 保持

正式共享模块：

```text
src/vrile/local_objects.py
```

至少包含：

```text
detect_connected_components(...)
merge_local_patches(...)
```

Stage 1 与 Stage 3 必须继续调用同一 shared primitive。

`detect_connected_components(...)` 保持 Stage 1 canonical semantics：

```text
finite/valid mask
SIC-change thresholding
8-neighbour connected-component labeling
min-cells filtering
component cell indices
component metrics
```

`merge_local_patches(...)` 保持 `consolidate_local_objects.py` canonical merge/membership semantics。

Batch 2R **不得修改 shared local-object scientific semantics**，尤其不得改：

```text
threshold
connectivity
binary closing
min-cells rule
component traversal order
Stage 1 weighted geographic centroid definition
merge_days
merge_distance_km
merge order
```

其中 Stage 1 weighted geographic centroid 即使存在 dateline 局限，本轮也只作为 legacy scientific semantics保留；Track S component geometry 在 `vrile.stage3` 内独立修复，不反向修改 Stage 1 primitive。

`footprint_strategy_audit.json` 只记录计划/架构状态：

```text
planned_strategy = Plan A
implementation_status = pending_batch2_reconciliation
```

不得在 local footprint step实际运行前预写 `completed`。最终完成状态由 reconciliation outputs 和 `batch2r_run_metadata.json` 确认。

### 7.2 Footprint reconstruction 必须使用 Stage 1 exact valid-domain semantics

Batch 2R backfill 必须复用 Stage 1 local detector 的 geolocation/mask path：

```text
SIC source dataset
→ vrile.io._add_projected_lat_lon
→ lon2 / lat2
→ parse_region("pan_arctic").contains(lon2, lat2)
→ valid = finite(diff) & pan_mask
```

不得再用：

```text
NSIDC-0780 codes 0–18 ocean mask
```

替代 Stage 1 local detector 的 `pan_arctic` mask。

NSIDC-0780 ocean domain继续用于 Stage 3 pan-Arctic accounting，但不是 Stage 1 footprint reconstruction mask。

调用 shared detector 时必须传：

```text
lon2
lat2
```

以真正重算 Stage 1 centroid/max-loss metrics。

### 7.3 Strict reconciliation：component correspondence + canonical object_id assignment

Stage 3 不独立“重新发明” legacy `object_id`。

对每个 canonical group：

```text
(date, start_date, window_days, threshold)
```

先执行 shared detector，要求：

```text
reported component count == recomputed component count
```

再按 Stage 1 canonical connected-component traversal order，与组内 `object_id` 顺序一一对应。

只有下列 multimetric fingerprint 全部通过后，才允许把 canonical Stage 1 `object_id/patch_key` 赋给 recomputed cell membership：

```text
component rank/order
cell_count
mean_sic_change
min_sic_change
cumulative_sic_loss
weighted centroid_lon
weighted centroid_lat
max_loss_lon
max_loss_lat
legacy nominal area_km2
```

输出：

```text
outputs/panarctic_local_contribution/
footprint_detector_reconciliation.csv
```

至少：

```text
group_key
reported_object_id
reported_patch_key
reported_component_rank
recomputed_label_id
recomputed_component_rank

reported_cell_count
recomputed_cell_count

reported_mean_sic_change
recomputed_mean_sic_change
reported_min_sic_change
recomputed_min_sic_change
reported_cumulative_sic_loss
recomputed_cumulative_sic_loss

reported_centroid_lon
recomputed_centroid_lon
reported_centroid_lat
recomputed_centroid_lat
reported_max_loss_lon
recomputed_max_loss_lon
reported_max_loss_lat
recomputed_max_loss_lat

reported_legacy_nominal_area_km2
recomputed_legacy_nominal_area_km2
recomputed_nsidc0771_cell_area_km2

status
```

Gate：

```text
all canonical patch rows covered
reported object_id set exact
group component count exact
component rank/order exact
cell_count exact
numeric metrics np.isclose at existing CSV precision
no systematic bias
```

`625 km²` 只允许出现在：

```text
legacy nominal area reconciliation
```

Stage 3 scientific integration继续使用 NSIDC-0771 v1.1 cell area。

删除当前 footprint reconstruction 中参数完全相同的 no-op fallback。任何 reconciliation FAIL → hard stop；不得复制第二套 detector。

### 7.4 Unique event ↔ member patch

继续使用 shared `merge_local_patches(...)` 按 canonical merge semantics恢复：

```text
unique_local_event_id
→ member object_id / patch_key
```

必须以 `outputs/local_vrile_enhanced/unique_local_events.csv` 的 actual canonical parameter fields 为准；不得混合 sensitivity configurations，也不得使用 region/date heuristic。

输出：

```text
data/processed/local_event_cells/
unique_event_patch_membership.csv
```

Event-level reconciliation：

```text
unique event count
unique_local_event_id
event_start
event_end
cumulative_loss
max_area
```

输出：

```text
outputs/panarctic_local_contribution/
unique_event_membership_reconciliation.csv
```

全部一致才 PASS。

### 7.5 Sparse footprint storage

```text
data/processed/local_event_cells/
├── <year>.npz
├── patch_cell_index.csv
└── unique_event_patch_membership.csv
```

每个 patch 至少可恢复：

```text
object_id / patch_key
date
start_date
window_days
threshold
y_idx
x_idx
delta_sic
```

不得为每个 local event 保存 full-Arctic dense raster。

## 8. Batch 2R spatial fields 与 QA

### 8.1 Track D：保留科学结果，增加 implementation cross-check

现有 Track D：

```text
99/99 internal field-budget closure PASS
D1a fidelity PASS
```

Batch 2R 不重新定义 Track D science。

保留：

```text
data/processed/panarctic_detector_extent_fields/<pan_event_id>.nc
outputs/panarctic_local_contribution/detector_extent_budget_closure_audit.csv
```

并新增逐事件 implementation cross-check：

```text
Track D detector_geotiff_delta_sie_km2 / 1e6
vs
foundation panarctic_detector_source_closure_audit.csv
    ::geotiff_reconstructed_delta_sie
```

按 `pan_event_id` join，99/99 必须 `np.isclose(rtol=1e-12, atol=1e-12)` PASS（comparison units = million km²）。

同时从现有 Track D audit 计算：

```text
detector_source_status_transition_activity_km2
=
detector_source_status_ice_loss_km2
+
detector_source_status_ice_gain_km2

detector_source_status_transition_activity_fraction
=
activity
/
(
detector_gross_extent_loss_km2
+
detector_gross_extent_gain_km2
+
activity
)
```

保留 event-level字段，并汇总：

```text
median
P90
max
n_events_gt_1pct
n_events_gt_5pct
```

该 summary 当前只作 QA，不新增 scientific hard threshold。

### 8.2 Track S：保持 component masks，修复 geometry

99 个 CDR field 继续使用：

```text
end_date   = unique event date
start_date = end_date - 5 days
```

严格复用 `reproduce_sic_locations.py` 5-day semantics。

保持既有：

```text
loss threshold
binary morphology
connectivity
min_object_cells
all valid loss components
```

不得因为 Batch 2R 重新识别另一套 components。

#### Component geometry

当前 `centroid_x/y` 与 `centroid_lon/lat` 不是同一个 centroid；Batch 2R 必须统一。

对每个 component 定义 area-consistent SIC-loss weight：

```text
w_i = cdr_sic_loss_i × cell_area_i
```

在 CDR projected grid 上：

```text
x_c = Σ(w_i x_i) / Σw_i
y_c = Σ(w_i y_i) / Σw_i
```

随后使用**实际 CDR SIC grid CRS metadata**将同一个 `(x_c, y_c)` transform 到 EPSG:4326。

不得：

```text
直接普通平均 longitude
把 unweighted x/y 与 weighted lon/lat 混为同一 centroid
硬编码错误的 source CRS
```

正式 component geometry字段：

```text
loss_area_weighted_centroid_x_m
loss_area_weighted_centroid_y_m
loss_area_weighted_centroid_lon
loss_area_weighted_centroid_lat
```

旧的模糊字段：

```text
centroid_x
centroid_y
centroid_lon
centroid_lat
```

不得继续作为 canonical output；优先删除，避免 Batch 3 误读。

`panarctic_loss_components.csv` 一行一个 component，至少：

```text
pan_event_id
component_id
cell_count
area_km2
integrated_sic_loss_km2eq
loss_area_weighted_centroid_x_m
loss_area_weighted_centroid_y_m
loss_area_weighted_centroid_lon
loss_area_weighted_centroid_lat
```

Component membership 可遵循 morphology；但 `integrated_sic_loss_km2eq` 始终积分真实 `cdr_sic_loss` 正值场，不给 morphology 填充单元伪造 loss。

#### Track S internal budget QA

每个 pan event计算：

```text
cdr_signed_sic_change_km2eq
=
Σ(cdr_signed_sic_change × cell_area)

cdr_gross_sic_loss_km2eq
=
Σ(cdr_sic_loss × cell_area)

cdr_gross_sic_gain_km2eq
=
Σ(cdr_sic_gain × cell_area)
```

Internal closure：

```text
cdr_gross_sic_gain_km2eq
-
cdr_gross_sic_loss_km2eq
≈
cdr_signed_sic_change_km2eq
```

Field-construction阶段继续检查：

```text
component row count
==
n_loss_components

0 <= component_resolved_sic_loss_fraction <= 1
```

但：

```text
resolved_sic_loss
-
Σ same-call component integrated_sic_loss
```

来自同一批in-memory values，只是内部加法一致性，不得再作为 independent component-loss closure，也不得在field-construction阶段据此写 `component_loss_closure_status = PASS`。

Track S signed/gross SIC internal budget closure仍使用 `np.isclose(rtol=1e-12, atol=1e-6 km²eq)`，99/99 event必须 PASS。正式component raster/table closure由Section 10.1重新打开persisted NetCDF后独立完成。

保存：

```text
data/processed/panarctic_loss_fields/<pan_event_id>.nc
```

字段保持：

```text
cdr_signed_sic_change
cdr_sic_loss
cdr_sic_gain
loss_component_id
x
y
```

Processed continuous fields继续以 `float32` 保存；`loss_component_id` 使用整数类型。为保证保存栅格与table/summary描述同一 processed field representation，component loss、component loss-weighted centroid weights、event gross loss/gain/signed integral与resolved loss必须从**实际将被保存的float32 arrays**转为float64后积分，并显式使用float64 accumulator。Component detection仍使用既有float64 detection path，不因serialization repair改变 threshold、morphology、connectivity、min-cell或component membership。

Field-construction写summary时，`component_loss_closure_error_km2eq` / `component_loss_closure_status` 尚未经过persisted-raster independent audit，应分别写为 `NaN` 与 `PENDING_INDEPENDENT_RECONCILIATION`（或严格等价的明确pending状态）；`stage3_cdr_loss_fields.py` 不得要求该pending字段PASS。Section 10.1 preflight是这两个字段的authoritative source，并在independent reconciliation后写回正式error/status。

Event summary：

```text
panarctic_loss_field_event_summary.csv
```

严格 99 行，至少：

```text
pan_event_id
event_date
cdr_window_start
cdr_window_end

cdr_signed_sic_change_km2eq
cdr_gross_sic_loss_km2eq
cdr_gross_sic_gain_km2eq
cdr_budget_closure_error_km2eq
cdr_budget_closure_status

resolved_sic_loss_km2eq
component_loss_closure_error_km2eq
component_loss_closure_status
component_resolved_sic_loss_fraction
n_loss_components
```

唯一 canonical 字段名：

```text
component_resolved_sic_loss_fraction
```

禁止使用：

```text
component_resolved_loss_fraction
```

后续 component-level N_eff 必须使用修复后的 geometry，并与 `component_resolved_sic_loss_fraction` 联合解释。

## 9. Batch 2R temporal alignment 与 raster overlap

### 9.1 Primary temporal alignment：patch-level + no-look-ahead

Pan-Arctic CDR window：

```text
[T-5, T]
T = primary pan-Arctic unique event date
```

Primary aligned patch 必须同时满足：

```text
patch.start_date <= T
patch.date       >= T-5
patch.date       <= T
```

即：

```text
patch interval intersects [T-5, T]
AND
patch end date <= T
```

最后一条是 no-look-ahead gate。Primary attribution footprint 不得使用 `patch.date > T` 的未来 SIC 信息。

Primary candidate ULE IDs 必须直接由：

```text
aligned_membership rows
```

派生。

不得再使用：

```text
ULE event_start/event_end overlap
→ take all lifetime member patches
```

作为 primary overlap source。

Primary local-event footprint：

> 仅 union 当前 pan event 的 `aligned_membership` 中属于该 ULE 的 patches。

不得使用 lifetime footprint。

### 9.2 Time-alignment repair audit

输出：

```text
outputs/panarctic_local_contribution/
time_alignment_audit.csv
```

严格 99 行，每个 pan event 至少：

```text
pan_event_id
event_date
pan_window_start
pan_window_end

old_ule_lifespan_candidate_count
patch_interval_candidate_count
primary_no_lookahead_candidate_count

old_lifetime_member_patch_count
interval_intersecting_patch_count
primary_no_lookahead_patch_count
future_patch_excluded_count

old_union_footprint_cells
primary_union_footprint_cells
union_footprint_cell_reduction
union_footprint_cell_reduction_fraction
```

其中：

```text
old_* = 复现初始 Batch 2 bug逻辑，仅用于 repair audit
patch_interval_* = patch interval intersects [T-5,T]，允许 patch.date>T，仅作中间诊断
primary_no_lookahead_* = 正式 scientific input
```

要求：

```text
primary_union_footprint_cells <= old_union_footprint_cells
future_patch_excluded_count >= 0
union_footprint_cell_reduction >= 0
```

最终汇总：

```text
n_events_affected
total_future_patches_excluded
median_union_footprint_cell_reduction_fraction
P90_union_footprint_cell_reduction_fraction
max_union_footprint_cell_reduction_fraction
```

### 9.3 Event level

每个 unique local event只分配一个 level：

```text
major_severe > severe > broad
```

即：

```text
if ULE in major_severe:
    major_severe
elif ULE in severe:
    severe
else:
    broad
```

不得重算 severity，不得重复 level。

### 9.4 Pan-event overlap summary：完整 99-event母表

输出：

```text
panarctic_event_overlap_summary.csv
```

严格 99 行、一行一个 `pan_event_id`。

所有 counts/union metrics 必须来自 **primary no-look-ahead aligned membership**。

至少：

```text
pan_event_id
event_date
n_temporal_candidates
no_temporal_candidate
n_broad_candidates
n_severe_candidates
n_major_severe_candidates

union_local_footprint_cells
union_local_footprint_area_km2
cdr_sic_loss_in_union_local_footprint_km2eq
cdr_sic_gain_in_union_local_footprint_km2eq
detector_gross_extent_loss_in_union_local_footprint_km2
detector_gross_extent_gain_in_union_local_footprint_km2
```

无 primary temporal candidate：

```text
n_temporal_candidates = 0
no_temporal_candidate = true
counts / union metrics = 0
```

Internal gates：

```text
n_temporal_candidates
=
n_broad_candidates
+
n_severe_candidates
+
n_major_severe_candidates

set(summary-derived candidate ULE IDs)
=
set(primary aligned membership ULE IDs)
```

### 9.5 Pan × local-event relation table

对 primary temporal candidates 输出：

```text
panarctic_local_event_overlap.csv
```

一行一个：

```text
pan event × unique local event
```

至少：

```text
pan_event_id
unique_local_event_id
event_level
time_aligned_patch_count
local_footprint_cells
local_footprint_area_km2
cdr_sic_loss_in_local_footprint_km2eq
cdr_sic_gain_in_local_footprint_km2eq
detector_gross_extent_loss_in_local_footprint_km2
detector_gross_extent_gain_in_local_footprint_km2
```

`time_aligned_patch_count` 必须严格等于该 `pan_event × ULE` 的 primary no-look-ahead aligned patch count。

即使 CDR component overlap=0，也保留真实 primary temporal-candidate relation row。

每个 pan event要求：

```text
set(relation-table ULE IDs)
=
set(primary aligned membership ULE IDs)
```

### 9.6 Component-level overlap

仅对：

```text
pan event
× primary aligned local event
× overlapping CDR loss component
```

输出：

```text
panarctic_local_component_overlap_pairs.csv
```

至少：

```text
pan_event_id
unique_local_event_id
event_level
component_id
intersection_cells
intersection_area_km2
IoU
local_capture_fraction
component_capture_fraction
cdr_sic_loss_in_intersection_km2eq
cdr_loss_weighted_overlap_fraction
```

Pair rows不直接做 additive summation，因为 local footprints 可重叠。

初始 Batch 2 baseline：

```text
pan×local rows = 2806
major_severe candidate rows = 382
component pairs = 5437
```

Batch 2R 完成后必须报告：

```text
2806 → new pan×local rows
382 → new major_severe rows
5437 → new component pairs
```

这些 baseline 仅用于 repair impact audit，不是结果必须接近旧值的 gate。

### 9.7 Batch 3 input boundary

Batch 2R overlap outputs的 primary temporal semantics是 Track S `[T-5,T]` no-look-ahead alignment。

因此：

```text
panarctic_local_event_overlap.csv
panarctic_local_component_overlap_pairs.csv
```

继续作为 Track S relation/component overlap input。

下列 Batch 2R columns：

```text
detector_gross_extent_loss_in_local_footprint_km2
detector_gross_extent_gain_in_local_footprint_km2
detector_gross_extent_loss_in_union_local_footprint_km2
detector_gross_extent_gain_in_union_local_footprint_km2
```

只保留为 cross-track diagnostic。

**Batch 3 Track D class budget必须从 sparse patch membership按 `[T-2,T+1]` 独立重建 `M_local_detector(T)`，不得读取上述 columns作为 exact detector-aligned attribution。**

## 10. Batch 3：preflight hardening

Batch 3 scientific metrics之前完成四项 gate。

### 10.1 Independent component raster/table reconciliation

现有 `component_loss_closure_error_km2eq` 不得继续用同一 accumulator与其 table rows自我求和验证。

正式 reconciliation 以保存后的 processed raster为 independent source。若 continuous fields使用float32保存，则 component table与event summary必须在 field-construction step基于相同serialized-precision arrays计算；不得通过放宽QA tolerance吸收serialization mismatch。

从保存的：

```text
data/processed/panarctic_loss_fields/<pan_event_id>.nc
```

独立计算：

```text
resolved_sic_loss_from_raster_km2eq
=
Σ(cdr_sic_loss × cell_area where loss_component_id > 0)
```

再与：

```text
Σ panarctic_loss_components.integrated_sic_loss_km2eq
```

比较。

逐 component还必须比较：

```text
raster cell_count
vs component-table cell_count

raster integrated SIC loss
vs component-table integrated_sic_loss_km2eq
```

输出：

```text
cdr_component_raster_table_reconciliation.csv
```

所有loss reconciliation统一使用：

```text
np.isclose(rtol=1e-12, atol=1e-6)
```

不得为float32 serialization另设宽松gate。

要求：

```text
3463/3463 component rows PASS
99/99 event resolved-loss closure PASS
```

并用independent raster result更新 `panarctic_loss_field_event_summary.csv` 的：

```text
resolved_sic_loss_km2eq
component_loss_closure_error_km2eq
component_loss_closure_status
component_resolved_sic_loss_fraction
```

Batch 3R只允许serialization-provenance repair；component count、component IDs或cell membership改变即HOLD并报告。

Batch 3R targeted validation必须在重生成前对99个persisted `loss_component_id` arrays按canonical int32 bytes记录SHA256 baseline，重生成后99/99 exact hash match。该hash只验证component IDs/cell membership没有因repair改变。

另保存修复前99-event的：

```text
resolved_sic_loss_km2eq
cdr_gross_sic_loss_km2eq
cdr_gross_sic_gain_km2eq
cdr_signed_sic_change_km2eq
```

并与修复后按 `pan_event_id` 比较，输出validation-only numeric-delta audit及median/P90/max absolute delta。这里不设通用scientific threshold；event-ID mismatch、non-finite delta或membership hash mismatch为hard failure，delta magnitude本身透明报告。

### 10.2 All-required CDR grid/CRS audit

扫描全部 unique required CDR dates：

```text
{T-5, T} for 99 pan-Arctic events
```

输出：

```text
cdr_required_grid_crs_signature_audit.csv
```

至少：

```text
date
file
shape
x/y grid signature
CRS signature
status
```

Per-file audit必须从**当前文件自身metadata**解析CRS。针对当前G02202/CDR实际schema，优先按SIC data variable的 `grid_mapping` attribute定位当前dataset内的grid-mapping variable；若当前schema确实缺少该link，只允许使用当前dataset自身存在且已确认的 `crs` grid-mapping variable。

对当前grid-mapping attrs分别尝试：

```text
valid CF single-property grid-mapping attrs
→ pyproj.CRS.from_cf(current attrs)
→ crs_cf

current-file spatial_ref / crs_wkt, if present
→ parse current-file WKT
→ crs_wkt
```

Authoritative per-file CRS规则：

```text
crs_cf parseable
→ authoritative CRS = crs_cf

crs_cf unparseable AND crs_wkt parseable
→ authoritative CRS = crs_wkt

both unparseable
→ FAIL

both parseable AND
crs_cf.equals(crs_wkt, ignore_axis_order=True) == false
→ current-file metadata conflict
→ FAIL / HOLD
```

即：CF single-property grid-mapping metadata可解析时优先；WKT是current-file fallback/consistency diagnostic，不得静默覆盖与CF attrs冲突的定义。

使用：

```text
from pyproj.enums import WktVersion

authoritative_crs.to_wkt(
    version=WktVersion.WKT2_2019,
    pretty=False,
)
```

生成canonical WKT并hash。不得调用representative `sic_crs()`生成per-file signature，也不得在当前文件CRS解析失败时fallback到representative file。

Grid signature使用shape/dimension order与完整canonical x/y coordinate arrays；x/y统一转little-endian float64 C-order bytes后hash，不得只比较first/last/step。

Audit至少记录CRS parse provenance / consistency：

```text
grid_mapping_var
crs_parse_source
cf_crs_parse_status
wkt_crs_parse_status
cf_wkt_consistency_status
crs_equals_reference
crs_equals_reference_ignore_axis_order
```

其中current-file `cf_wkt_consistency_status` 至少区分：

```text
CONSISTENT
CF_ONLY
WKT_ONLY
CONFLICT
UNPARSEABLE
```

`crs_equals_reference*` 只作reference semantic-equivalence diagnostic，不替代signature gate。

Gate：

```text
all 197 required files readable
unique full-grid signature = 1
unique actual per-file CRS signature = 1
```

若任一current-file CRS不可解析、CF/WKT定义发生semantic conflict，或CRS signature不唯一，Batch 3R HOLD并报告signature counts、parse source/consistency status与semantic-equivalence diagnostics；不得自动collapse semantically equivalent CRS。通过后才允许其他已验证shared geometry path继续使用representative `sic_crs()`。

### 10.3 D1a audit required

当：

```text
route == exact_detector_attribution
```

以下文件必须存在且 gate仍 PASS：

```text
panarctic_detector_source_closure_audit.csv
panarctic_detector_source_closure_summary.csv
```

不得 conditional silent skip。

### 10.4 Provenance refresh

更新：

```text
footprint_strategy_audit.json
stage3_method_contract.json
```

至少记录：

```text
footprint implementation_status = verified_batch2r_reconciliation
verified_patch_count = 35559
Stage 3 scope includes Batch 2R PASS and current Batch 3R acceptance repair
dual_track_alignment_locked = true
```

上述为 provenance，不改变 Stage 1/2 scientific outputs。

---

## 11. Batch 3：dual-track class-union contribution budget

Event hierarchy仍为：

```text
major_severe > severe > broad
```

每个 ULE只赋一个 level。

### 11.1 Track-specific aligned membership

Track D：

```text
patch.start_date <= T+1
patch.date       >= T-2
patch.date       <= T+1
```

Track S：

```text
patch.start_date <= T
patch.date       >= T-5
patch.date       <= T
```

直接从 aligned membership派生 candidate ULE IDs和 patch IDs。

输出：

```text
track_alignment_audit.csv
```

99 行，至少：

```text
pan_event_id
detector_aligned_patch_count
detector_aligned_ule_count
cdr_aligned_patch_count
cdr_aligned_ule_count
shared_ule_count
detector_only_ule_count
cdr_only_ule_count
```

### 11.2 Exclusive class partition

对每个 Track先构建 raw class unions：

```text
M_major_raw
M_severe_raw
M_broad_raw
```

再按优先级生成单值 class code：

```text
0 residual / no aligned local footprint
1 broad_nonsevere
2 severe_nonmajor
3 major_severe
```

等价于：

```text
major_severe    = M_major_raw
severe_nonmajor = M_severe_raw & ~M_major_raw
broad_nonsevere = M_broad_raw & ~(M_major_raw | M_severe_raw)
residual        = complement
```

保存：

```text
data/processed/panarctic_class_partition_fields/<pan_event_id>.nc
```

至少：

```text
detector_local_class_code
cdr_local_class_code
x
y
```

attrs写 class-code book与两套 temporal semantics。

输出：

```text
class_partition_overlap_audit.csv
```

每 event / track至少记录：

```text
raw_major_cells
raw_severe_cells
raw_broad_cells
major_severe_overlap_cells
major_broad_overlap_cells
severe_broad_overlap_cells
triple_overlap_cells
raw_cross_class_overlap_cells
raw_union_cells
raw_cross_class_overlap_fraction
exclusive_major_severe_cells
exclusive_severe_nonmajor_cells
exclusive_broad_nonsevere_cells
residual_cells
expected_partition_exact_match
status
```

定义：

```text
raw_cross_class_overlap_cells
= cells covered by at least two raw class unions

raw_union_cells
= cells covered by at least one raw class union

raw_cross_class_overlap_fraction
= raw_cross_class_overlap_cells / raw_union_cells
```

`raw_union_cells == 0` 时fraction记为0。

Production class-code path与audit reconstruction path必须分开：production path生成并保存class code；audit path从三个raw boolean unions按上述priority公式独立构建 `expected_class_code`，随后重新打开保存的NetCDF并执行：

```text
np.array_equal(expected_class_code, persisted_class_code)
```

不得把同一array与自身比较并称为independent gate。

Priority partition必须 mutually exclusive / exhaustive；class codes仅允许0/1/2/3。Missing aligned patch cells必须hard fail，不得silent skip。

### 11.3 Track D contribution budget

仅在：

```text
detector_extent_loss_mask
detector_extent_gain_mask
```

上按 `detector_local_class_code` 统计 physical extent。

输出：

```text
panarctic_detector_contribution_budget.csv
```

一行一个 pan event，至少：

```text
detector_gross_extent_loss_km2
detector_gross_extent_gain_km2
detector_physical_net_extent_change_km2

detector_extent_loss_major_severe_km2
detector_extent_loss_severe_nonmajor_km2
detector_extent_loss_broad_nonsevere_km2
detector_extent_loss_residual_km2

detector_extent_gain_major_severe_km2
detector_extent_gain_severe_nonmajor_km2
detector_extent_gain_broad_nonsevere_km2
detector_extent_gain_residual_km2

major_detector_extent_loss_fraction
severe_nonmajor_detector_extent_loss_fraction
broad_nonsevere_detector_extent_loss_fraction
residual_detector_extent_loss_fraction

detector_loss_partition_closure_error_km2
detector_gain_partition_closure_error_km2
detector_partition_closure_status

detector_compensation_ratio
detector_net_loss_efficiency
```

定义：

```text
compensation_ratio = G / L
net_loss_efficiency = (L-G) / L
```

不 clip。

Track D primary fractions的科学表述：

> fraction of detector-aligned G02135 gross physical extent loss spatially covered by the exclusive aligned local-event class footprint.

不得写 causal contribution fraction。

### 11.4 Track S contribution budget

在 full valid-ocean CDR field上按 `cdr_local_class_code` 积分：

```text
cdr_sic_loss × cell_area
cdr_sic_gain × cell_area
```

输出：

```text
panarctic_cdr_contribution_budget.csv
```

至少：

```text
cdr_gross_sic_loss_km2eq
cdr_gross_sic_gain_km2eq
cdr_signed_sic_change_km2eq

cdr_sic_loss_major_severe_km2eq
cdr_sic_loss_severe_nonmajor_km2eq
cdr_sic_loss_broad_nonsevere_km2eq
cdr_sic_loss_residual_km2eq

cdr_sic_gain_major_severe_km2eq
cdr_sic_gain_severe_nonmajor_km2eq
cdr_sic_gain_broad_nonsevere_km2eq
cdr_sic_gain_residual_km2eq

major_cdr_sic_loss_fraction
severe_nonmajor_cdr_sic_loss_fraction
broad_nonsevere_cdr_sic_loss_fraction
residual_cdr_sic_loss_fraction

cdr_loss_partition_closure_error_km2eq
cdr_gain_partition_closure_error_km2eq
cdr_partition_closure_status

cdr_compensation_ratio
cdr_net_loss_efficiency
```

Track S fractions是 CDR continuous SIC-loss composition，不得称 exact G02135 attribution。

### 11.5 Partition gates

99/99 events必须：

```text
Track D class loss sum ≈ detector gross physical loss
Track D class gain sum ≈ detector gross physical gain
Track S class SIC-loss sum ≈ CDR gross SIC loss
Track S class SIC-gain sum ≈ CDR gross SIC gain
loss fractions sum ≈ 1 when denominator > 0
all class amounts >= 0
```

使用：

```text
np.isclose(rtol=1e-12, atol=1e-6)
```

任一 FAIL → stop。

### 11.6 Canonical schema gate

Batch 3尚未冻结，因此canonical outputs不得长期同时保留generic scientific aliases。

对：

```text
panarctic_detector_contribution_budget.csv
panarctic_cdr_contribution_budget.csv
panarctic_spatial_composition_metrics.csv
```

分别定义required-column set与forbidden-generic-alias set。Gate必须同时满足：

```text
99 unique pan_event_id
event ID set == primary 99 pan events
missing_required == empty
forbidden_present == empty
```

至少禁止当前已确认的generic drift names：

```text
gross_loss
gross_gain
residual_gross_loss
broad_nonsevere_gross_loss
severe_nonmajor_gross_loss
major_severe_gross_loss
partition_loss_closure_error
partition_gain_closure_error
partition_closure_status
compensation_ratio
net_loss_efficiency
largest_component_resolved_loss_share
top3_component_resolved_loss_share
```

不得仅靠数值closure判定schema PASS，也不得在canonical columns旁长期保留一套旧generic scientific columns作为兼容层。

---

## 12. Batch 3：multi-center 与 regional composition metrics

### 12.1 Component scale：Q1 primary

基于：

```text
panarctic_loss_components.csv
```

对每 event：

```text
p_j = component_loss_j / resolved_sic_loss
N_eff = 1 / Σ(p_j²)
```

所有 component shares以 **resolved component loss** 为 denominator。

输出 `panarctic_spatial_composition_metrics.csv`，至少：

```text
effective_component_number_sic
largest_component_resolved_sic_loss_share
top3_component_resolved_sic_loss_share
n_loss_components
top2_component_centroid_distance_km
weighted_component_spatial_spread_km
component_resolved_sic_loss_fraction
```

`top2_component_centroid_distance_km`：

```text
two largest-loss component centroids
→ pyproj.Geod WGS84 geodesic distance
```

`weighted_component_spatial_spread_km`：

```text
component-loss-weighted spherical centroid
→ WGS84 geodesic distance of each component centroid to that center
→ weighted RMS distance
```

不得普通平均 longitude。

Canonical output必须包含 `pan_event_id,event_date`。Largest/top3 canonical字段名固定为：

```text
largest_component_resolved_sic_loss_share
top3_component_resolved_sic_loss_share
```

不得缩写为generic `*_resolved_loss_share`。

N_eff / largest/top3 share必须与 `component_resolved_sic_loss_fraction` 联合解释。

### 12.2 Region budgets

使用 NSIDC-0780 codes 0–18。

Code 0：

```text
non_region_ocean
is_named_region = false
```

Codes 1–18：

```text
named physical regions
```

输出：

```text
panarctic_detector_region_budget.csv
panarctic_cdr_region_budget.csv
```

每个 pan event × region一行，分别记录 Track D extent loss/gain和 Track S SIC loss/gain。

Region N_eff只对 named regions 1–18计算：

```text
p_r = named-region loss_r / Σ named-region loss
N_eff_region = 1 / Σ(p_r²)
```

在 `panarctic_spatial_composition_metrics.csv` 追加：

```text
effective_named_region_number_extent_loss
effective_named_region_number_sic_loss

largest_named_region_extent_loss_share
top3_named_region_extent_loss_share
largest_named_region_sic_loss_share
top3_named_region_sic_loss_share

non_region_ocean_extent_loss_fraction
non_region_ocean_sic_loss_fraction
```

Named-region shares的 denominator是 codes 1–18 named-region total loss。

Code 0 fraction的 denominator必须是对应 Track 的pan-Arctic global total loss，并从canonical contribution budget按 `pan_event_id` join：

```text
non_region_ocean_extent_loss_fraction
= code0 detector loss / detector_gross_extent_loss_km2

non_region_ocean_sic_loss_fraction
= code0 CDR SIC loss / cdr_gross_sic_loss_km2eq
```

不得使用 `Σ codes 0–18 loss` 作为code 0 denominator。

由于Track D physical transition field与NSIDC-0780 codes 0–18 accounting domain并非先验完全相同，Batch 3R额外输出：

```text
region_budget_accounting_audit.csv
```

严格99行，至少记录两个Track的global loss、codes 0–18 accounted loss、unassigned loss与unassigned fraction。该表仅作surface-mask accounting QA；non-zero unassigned不自动解释为scientific failure，也不得把其他surface-mask codes重新解释为code 0。

Batch 3R在修改canonical schema/denominator前，应从当前Track D global budget与region budget逐event记录pre-repair unassigned amount/fraction baseline。由于本轮不得改变Track D detector field、physical-transition semantics或region-budget amounts，修复后的Track D accounting gap必须与baseline逐event回归一致；amount使用 `np.isclose(rtol=1e-12, atol=1e-6)`，fraction使用 `np.isclose(rtol=1e-12, atol=1e-12)`。这是unrelated-numeric-change regression gate，不是“unassigned fraction必须低于某阈值”的science gate。

Track S accounting gap继续记录。基于当前valid-ocean construction预期接近numerical zero，但不得把该预期替换成新的scientific threshold。

这些指标描述 regional composition，不作为 component-scale multi-center primary metric。

---

## 13. Batch 4：matched controls、co-loss 与 spatial compensation

Batch 4采用分批实施：

```text
Batch 4A  Track S matched controls / residual and other-region co-loss / spatial compensation
Batch 4B  strong-local-event reverse analysis
Batch 4C  kinematic area-budget diagnostic
```

当前先实施 Batch 4A。

### 13.1 Track boundary

Batch 4A primary matched-control inference只使用 Track S / G02202 CDR 5-day fields。

原因：

```text
event/control window = [anchor-5, anchor]
```

而当前 G02135 GeoTIFF inventory只覆盖99个pan-Arctic event所需 detector dates，不覆盖背景control windows。

因此：

```text
Track D = frozen event-level exact detector attribution；本批不扩展到control inference
Track S = matched-control anomaly inference
```

不得为Batch 4A临时下载完整G02135 archive，也不得把CDR control result写成exact detector contribution。

### 13.2 Event-specific domains

对每个pan event，读取persisted：

```text
data/processed/panarctic_class_partition_fields/<pan_event_id>.nc
```

Track S major mask：

```text
M_major(T) = cdr_local_class_code == 3
```

Residual domain：

```text
valid_ocean_mask & ~M_major(T)
```

注意：这不是 Batch 3 class code 0 residual；它只排除同期 Track S aligned `major_severe` footprint，因此仍包含 severe_nonmajor、broad_nonsevere和class-0 cells。

对该event的所有controls使用同一个 event-specific `M_major(T)`；不得在control window重新识别或mask当时的local events。

`source_major_regions(T)`：

1. 复用 Batch 3 Track S `[T-5,T]` alignment；
2. 取aligned `major_severe` ULE IDs；
3. join `major_severe_events_union.csv` 的 canonical `dominant_region`；
4. 映射到现有NSIDC-0780 named region codes 1–18。

Unknown/ambiguous region或无法一一映射 → HOLD；不得从centroid猜region。

Other-region domain：

```text
named region codes 1–18
-
source_major_regions(T)
```

Code 0单独记账，不进入other-region domain。

若event无aligned major_severe ULE：

```text
source_major_regions = empty
other-region domain = all named regions 1–18
```

### 13.3 Matched controls

Event anchor为primary pan-Arctic unique event date `T`；event CDR window：

```text
[T-5,T]
```

Control anchor `t` 必须：

```text
same year as T
t in JJA
CDR files for t-5 and t readable
control window [t-5,t] does not intersect any of the 99 primary pan-event CDR windows [Ti-5,Ti]
```

匹配状态：

```text
initial SIE event   = SIE(T-5)
initial SIE control = SIE(t-5)
```

SIE使用现有 canonical processed Sea Ice Index daily table；不得从CDR重定义initial SIE。

搜索tier：

```text
|t-T| <= 30 days
then <= 45
then <= 60 days
```

选择最小的、候选数达到20的tier；在该tier内按：

```text
absolute initial-SIE difference
then control anchor date
```

排序取前20。

若±60 days：

```text
>=20 controls → matched_controls
10–19 controls → limited_background_controls，使用全部
<10 controls → insufficient_background_controls
```

不得引入非JJA control anchor，不强制排除control window中的local `major_severe` events。

每个control anchor可被不同pan events复用；control rows不是独立统计单位。

### 13.4 Event/control metrics

CDR field统一使用：

```text
signed = C_end - C_start
loss   = max(0, C_start - C_end)
gain   = max(0, C_end - C_start)
```

使用 frozen Track S grid、valid-ocean mask和NSIDC-0771 cell area。

每个event及其controls计算：

```text
global_sic_loss_km2eq
global_sic_gain_km2eq
global_compensation_ratio

residual_sic_loss_km2eq
residual_sic_gain_km2eq
residual_net_sic_loss_km2eq
residual_compensation_ratio

other_region_sic_loss_km2eq
other_region_sic_gain_km2eq
other_region_net_sic_loss_km2eq
other_region_compensation_ratio
```

定义：

```text
net_sic_loss = loss - gain
compensation_ratio = gain / loss
```

不clip；loss=0时ratio=NaN。

Event metrics和control metrics必须走同一scientific function。

### 13.5 Event-level differences

对每个 `pan_event_id × metric`：

```text
event_value
control_median
difference = event_value - control_median
```

只有event value finite且至少10个selected controls的该metric finite时，event进入该metric inference。

统计单位：

```text
pan_event_id
```

每个event每个metric只贡献一个difference。不得把control rows、regions或grid cells当独立样本。

### 13.6 Regional co-loss matrix与synchronization

Named region codes 1–18。

每个：

```text
pan_event_id × region_code
```

记录：

```text
event_sic_loss_km2eq
control_median_sic_loss_km2eq
loss_anomaly_km2eq
background_percentile
n_valid_controls
status
```

定义：

```text
loss_anomaly = event - control median
background_percentile = mean(control_loss <= event_loss)
```

严格输出99 × 18 rows；insufficient controls保留row并写status/NaN，不得删row。

每event描述性synchronization metrics：

```text
n_regions_above_control_median_sic_loss
fraction_regions_above_control_median_sic_loss
median_region_background_percentile
```

“above control median”是descriptive matched-background diagnostic，不称region-level significance。

Batch 4A不对18个region分别做显著性搜索，不生成`any(raw_p < 0.05)`类region gate。

### 13.7 Inference与FDR

Primary inference只对event-level differences。

Family 1：

```text
coloss
- residual_sic_loss_km2eq
- other_region_sic_loss_km2eq
```

方向性假说为event loss增强；使用event-unit one-sided positive sign-flip test。

Family 2：

```text
compensation
- global_compensation_ratio
- residual_compensation_ratio
- other_region_compensation_ratio
```

使用event-unit two-sided sign-flip test。

每metric：

```text
bootstrap mean-difference CI
p_raw
```

Bootstrap只resample event-level differences。

分别在两个预定义family内使用Benjamini-Hochberg FDR；保留raw与corrected p-values。

Status：

```text
n_events < 10            → insufficient_samples
p_fdr < 0.05             → supported
otherwise                → not_supported
```

Sign-flip不是新的独立evidence dimension；不得重复计分。

固定seed并记录resample/permutation counts。

### 13.8 Batch 4A outputs

```text
outputs/panarctic_local_contribution/
├── panarctic_matched_control_anchors.csv
├── panarctic_control_window_metrics.csv
├── panarctic_coloss_event_differences.csv
├── panarctic_regional_coloss_matrix.csv
├── panarctic_synchronization_metrics.csv
├── panarctic_control_test_summary.csv
└── coloss_control_run_metadata.json
```

生产文件名保持version-neutral。

Batch 4A完成后，再单独设计/实施 Batch 4B strong-local reverse analysis；不得直接复用pan-event inference unit或control design而不重新定义reverse-analysis unit。

---

## 14. Batch 4C：kinematic area-budget diagnostic

在 NSIDC ice-motion event-window coverage audit通过后实施。

目的：

> 区分 observed SIC tendency 中与 sea-ice kinematic transport一致的部分，以及 unresolved/non-kinematic residual。

基本关系：

\[
\frac{\partial C}{\partial t}
=
-\nabla\cdot(C\mathbf{u}_i)
+
R
\]

诊断：

```text
kinematic transport = -∇·(C u_i)
advection term       = -u_i·∇C
divergence term      = -C∇·u_i
unresolved residual  = observed tendency - kinematic transport
```

`R` 不得命名为 thermodynamic melt。

Primary domain使用 fixed Track S time-aligned local-event footprint/control volume；不以100/300/500 km buffer替代。

若 finite-volume implementation可靠，再做 boundary ice-area flux与transport-budget closure。

该模块是 mechanism diagnostic，不进入 Track D exact attribution。

---

## 15. Batch 5：supplementary / reconstruction / phenotype

Batch 5 可包含：

```text
masked-reconstruction diagnostic
PIOMAS area-volume decoupling
CDR concordance stratified QA
composition feature assembly
exploratory phenotype/clustering
scientific summary
```

Masked reconstruction不得称 counterfactual simulation。

PIOMAS volume tendency不得当作直接观测。

Phenotype命名必须来自实际 continuous feature / cluster characteristics，不预设故事阈值。

---

## 16. 实施批次

### Batch 0–1R2：已完成

```text
method semantics
cell-area/CRS alignment
G02135 inventory
D1a closure
CDR concordance
route decision
```

### Batch 2 initial：历史结果，overlap layer superseded

有效基础：

```text
Track D fields
shared local-object primitives
99 CDR fields
3463 component masks
```

旧 lifetime-footprint overlap结果不得作为 scientific input。

### Batch 2R：已完成，PASS

```text
patch-level no-look-ahead Track S alignment
time_alignment_audit
35559/35559 strict footprint reconciliation
unique-event membership reconciliation
Track D vs D1a cross-check
source-status activity QA
loss×area weighted component geometry
Track S SIC budget closure
recomputed Track S overlap
```

正式结果：

```text
2806 pan×local rows
382 major_severe rows
4464 component pairs
```

### Batch 3：已完成并冻结，PASS

```text
Batch 3R3 final acceptance PASS
dual-track alignment locked
strict persisted raster/table reconciliation PASS
persisted class partition reconstruction PASS
Track D / Track S contribution budgets frozen
component / regional composition metrics frozen
197-file CDR grid/CRS diagnosis completed
full Stage 3 run PASS
```

### Batch 4A：当前阶段

```text
Track S event-specific residual / other-region domains
same-year JJA 5-day matched controls
initial-SIE state matching
event-level co-loss differences
regional co-loss matrix
descriptive synchronization metrics
spatial compensation differences
event-unit sign-flip + bootstrap
explicit BH-FDR families
```

### Batch 4B：后续

```text
strong-local-event reverse analysis
```

必须单独定义reverse-analysis statistical unit、non-pan-event sample和background/control contract；不得直接复用Batch 4A pan_event_id inference。

### Batch 4C：后续，coverage gate

```text
kinematic area-budget diagnostic
```

### Batch 5

```text
masked reconstruction
supplementary area-volume decoupling
CDR concordance stratified QA
composition features
exploratory phenotype
scientific summary
```

Stage 3 core science稳定后重跑 StageX。

---

## 17. 核心输出

Foundation：

```text
outputs/panarctic_local_contribution/
├── stage3_method_contract.json
├── input_schema_audit.csv
├── surface_mask_accounting_audit.csv
├── cell_area_alignment_audit.csv
├── g02135_required_file_inventory.csv
├── panarctic_detector_source_closure_audit.csv
├── panarctic_detector_source_closure_summary.csv
├── panarctic_cdr_detector_concordance_audit.csv
├── panarctic_cdr_detector_concordance_summary.csv
└── stage3_route_decision.json
```

Batch 2 / 2R：

```text
outputs/panarctic_local_contribution/
├── g02135_geotiff_grid_signature_audit.csv
├── detector_extent_budget_closure_audit.csv
├── footprint_detector_reconciliation.csv
├── unique_event_membership_reconciliation.csv
├── panarctic_loss_components.csv
├── panarctic_loss_field_event_summary.csv
├── time_alignment_audit.csv
├── panarctic_event_overlap_summary.csv
├── panarctic_local_event_overlap.csv
├── panarctic_local_component_overlap_pairs.csv
├── batch2_run_metadata.json
└── batch2r_run_metadata.json
```

Batch 3：

```text
outputs/panarctic_local_contribution/
├── cdr_required_grid_crs_signature_audit.csv
├── cdr_component_raster_table_reconciliation.csv
├── track_alignment_audit.csv
├── class_partition_overlap_audit.csv
├── panarctic_detector_contribution_budget.csv
├── panarctic_cdr_contribution_budget.csv
├── panarctic_detector_region_budget.csv
├── panarctic_cdr_region_budget.csv
├── region_budget_accounting_audit.csv
├── panarctic_spatial_composition_metrics.csv
└── batch3_run_metadata.json
```

Batch 4A：

```text
outputs/panarctic_local_contribution/
├── panarctic_matched_control_anchors.csv
├── panarctic_control_window_metrics.csv
├── panarctic_coloss_event_differences.csv
├── panarctic_regional_coloss_matrix.csv
├── panarctic_synchronization_metrics.csv
├── panarctic_control_test_summary.csv
└── coloss_control_run_metadata.json
```

Processed spatial data：

```text
data/processed/panarctic_detector_extent_fields/<pan_event_id>.nc
data/processed/panarctic_loss_fields/<pan_event_id>.nc
data/processed/panarctic_class_partition_fields/<pan_event_id>.nc

data/processed/local_event_cells/
├── <year>.npz
├── patch_cell_index.csv
└── unique_event_patch_membership.csv
```

---

## 18. Scientific target

Track D：

> **Quantify detector-aligned G02135 extent loss associated with local rapid sea-ice loss footprints during pan-Arctic VRILEs.**

Track S：

> **Quantify CDR-based spatial SIC-loss composition and multi-center structure associated with Sea Ice Index-detected pan-Arctic VRILEs.**

Integrated Stage 3：

> **Quantify how geographically distributed local rapid sea-ice losses overlap with detector-aligned pan-Arctic extent loss, aggregate within the broader SIC-loss field, and are amplified or spatially compensated during pan-Arctic VRILEs.**
