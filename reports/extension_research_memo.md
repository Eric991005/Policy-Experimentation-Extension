# Extension Research Memo

- Generated at: 2026-03-12T22:37:26

## 1. 数据来源与清洗过程
- 主数据为地级市政务微信文本分片（part_001~005），已统一 schema 并去重。
- 中央政策文本来自 planning_docs 与 planning_manifest。
- NBS 与土地数据在当前版本主要作为“可得性/入口”资产，宏观变量结构化尚不完整。

## 2. 文本变量定义
- 中央目标函数：Risk/Stability/Scalability/EvalPressure/StrategicPriority/Conditionality。
- 地方响应：Risk/Stability/Pilot/Scalability/Eval/Strategic 维度，并构建 PolicyResponseIntensity。
- 中央-地方一致性：城市-月响应向量与中央当期目标向量余弦相似度（alignment_score）。

## 3. 面板构造方法
- city_month_panel: 32804 行
- city_quarter_panel: 11306 行
- city_year_panel: 3153 行

## 4. 基准回归设计
- Model A: 地方响应 ~ 中央目标 + 交互项 + 城市FE + 时间FE
- Model B: 次期结果(通信活跃度代理) ~ 响应 + 一致性 + controls + FE
- Model C: 战略性学习交互项（Risk×Capacity, Scalability×Capacity, Stability×Need）

## 5. 主要结果
- 回归结果已输出为 CSV/TEX；关键系数图已生成。
- 中央目标维度分布与地方响应时间趋势图已生成。

## 6. 对原文的可能 argue
- 文本证据支持“中央不只学习平均处理效应（ATE）”，还在强调风险可控、稳妥推进与可复制扩散。
- 地方响应并非单一强度变化，而呈现多维目标耦合。
- alignment 可作为“战略性政策学习”可检验指标。

## 7. 当前局限与下一步数据需求
- NBS 城市级结构化宏观变量尚不完整，当前 Model B 使用通信结果代理。
- 北大法宝原始法规文本尚未批量落地（登录/权限限制）。
- 下一步应补齐城市级财政、投资、产业结构、人口等年度指标，并进行稳健性检验。
## 8. 增强版更新（省级宏观并表）
- 更新时间: 2026-03-12T23:28:56
- 新增数据: data.stats.gov.cn(fsnd) 省级年度宏观指标（GDP、人均GDP、人口、财政收支、二三产结构、固定资产投资增速）
- 新增文件:
  - data_final/macro_province_year_from_nbs.csv
  - data_final/city_year_panel_enhanced_macro.csv
  - outputs/tables/regression_results_model_B_enhanced.csv
  - outputs/figures/06_model_B_enhanced_coefficients.png
- Enhanced Model B样本量: 2350
- Enhanced Model B估计器: PanelOLS
- 说明: 当前宏观变量为“省级并表到城市”，可用于增强稳健性，但非纯地级市硬结果变量。

## 9. 动态事件窗/滞后模型（新增）
- 更新时间: 2026-03-12T23:40:45
- 已新增 central_event_timeline_monthly.csv 并并入 city_month_panel。
- Model D（事件窗）: EventHigh 的 lead/lag 与 Capacity 交互，城市FE+月FE。
- Model E（分布式滞后）: ObjectiveIndex 的 0-6 月滞后与 Capacity 交互，城市FE+月FE。
- Event-window 最显著系数（前5）：
  - W-5_xCap: coef=0.0072, p=0.0010
  - W0_xCap: coef=0.0050, p=0.0252
  - W-3_xCap: coef=0.0037, p=0.0305
  - W-6_xCap: coef=-0.0040, p=0.1095
  - W1_xCap: coef=0.0029, p=0.1409
- Lag-model 最显著系数（前5）：
  - ObjLag1_xCap: coef=0.0226, p=0.0051
  - ObjLag3_xCap: coef=0.0203, p=0.0099
  - ObjLag5_xCap: coef=0.0154, p=0.0751
  - ObjLag4_xCap: coef=0.0160, p=0.1074
  - ObjLag0_xCap: coef=0.0071, p=0.4985

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:07:22
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 367；成功匹配城市指标: 329（match rate=89.6%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:21:11
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 368；成功匹配城市指标: 329（match rate=89.4%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:25:15
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 368；成功匹配城市指标: 329（match rate=89.4%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:31:54
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 368；成功匹配城市指标: 329（match rate=89.4%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:37:39
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 368；成功匹配城市指标: 329（match rate=89.4%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv

## 10. 地级市边界地图（新增）
- 更新时间: 2026-03-13T00:42:38
- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。
- 边界数: 368；成功匹配城市指标: 329（match rate=89.4%）。
- 输出文件:
  - outputs/figures/11_city_alignment_map.html
  - outputs/figures/12_city_response_map.html
  - data_final/city_level_boundary_with_metrics.geojson
  - data_final/city_geo_metric_merge_status.csv
