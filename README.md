# Coastal Property Climate Risk Pricing<br>海岸房产气候风险定价模型

> **How much does coastal erosion risk discount a property's market price?**  
> A machine learning approach to quantifying environmental risk in mortgage portfolios.
>
> **海岸侵蚀风险对房产市价的折价效应有多大？**  
> 基于机器学习的按揭贷款组合环境风险量化研究。

---

## Why This Project Matters · 项目价值

**EN:** British banks hold billions in mortgage assets along England's eroding coastline. As climate change accelerates coastal retreat, properties near erosion zones face declining values, rising insurance costs, and reduced mortgage availability. If left unmeasured, this environmental risk premium becomes a systemic threat to lenders.

This project simulates the work of a **risk management analyst at a UK commercial bank**. The brief: build a predictive model that estimates the *true* market price of a coastal property after accounting for erosion risk, so the bank can identify overvalued assets in its mortgage book before they become bad debts.

**核心问题 / Core Question:** For every metre closer to a predicted erosion zone, what discount does the market already price in? · 距离侵蚀区每近一米，市场已经为其定价了多少折价？

---

**中文：** 英国银行在英格兰侵蚀海岸线沿线持有的按揭资产高达数十亿英镑。随着气候变化加速海岸线退缩，侵蚀区附近的房产正面临价值缩水、保险费飙升和贷款获批难度加大的三重打击。如果不对这一环境风险溢价加以量化，它将成为银行体系的系统性威胁。

本项目模拟**英国商业银行风险管理分析师**的真实工作场景。任务目标是：构建一个预测模型，在扣除海岸侵蚀风险后估算房产的*真实*市场价，帮助银行在按揭资产变成坏账之前，识别出那些被高估的高风险资产。

---

## What This Project Demonstrates · 项目能力展示

### End-to-End Data Science Workflow · 全流程数据科学工作流

| Stage · 阶段 | What I Did · 工作内容 |
|---|---|
| **Data Integration**<br>数据集成 | EN: Merged 3 UK government open datasets (HM Land Registry, Ordnance Survey NSPL, Defra NCERM) totaling ~1.7M transaction records<br>中文：融合3个英国政府开放数据集（土地注册处、地形测量局、环境部），总计约170万条交易记录 |
| **Spatial Engineering**<br>空间工程 | EN: Broke a cross-agency data silo by converting 800K+ string postcodes into geographic coordinates, then computing metre-level distances to erosion boundaries<br>中文：打破跨部门数据孤岛，将80万+字符串邮编转化为地理坐标，并计算每处房产距离侵蚀边界的米级精确距离 |
| **Data Cleaning**<br>数据清洗 | EN: Diagnosed and handled real-world messiness: headerless CSVs, terminated postcodes (53% join loss with bias analysis), £1 symbolic transfers, right-skewed price distributions<br>中文：诊断并处理真实世界的脏数据——无表头CSV、已停用邮编（53%连接损失及偏差分析）、1英镑象征性转让、房价右偏态分布 |
| **Feature Engineering**<br>特征工程 | EN: Selected predictive features while removing high-cardinality ID-leakage columns (door numbers, postcodes) that cause memorisation<br>中文：筛选预测性特征，同时显式剔除会导致模型死记硬背的高基数ID泄露字段（门牌号、独立邮编） |
| **Exploratory Analysis**<br>探索性分析 | EN: Produced 5 publication-quality visualisations confirming the non-linear, threshold-based relationship between erosion proximity and price<br>中文：产出5张发表级可视化图表，证实侵蚀距离与房价之间存在非线性、存在阈值效应的关系 |
| **Model Iteration**<br>模型迭代 | EN: Progressed from naive baselines (LinearRegression, shallow DecisionTree) through ensemble methods (RandomForest, XGBoost) with rigorous statistical justification at each step<br>中文：从朴素基线模型（线性回归、浅层决策树）逐步迭代到集成方法（随机森林、XGBoost），每一步均有严谨的统计论证 |
| **Pipeline Engineering**<br>流水线工程 | EN: Built a production-ready `sklearn.pipeline.Pipeline` with `GridSearchCV` — the workflow a data scientist hands to a credit risk team<br>中文：构建可部署的 sklearn 流水线与 GridSearchCV 超参数网格搜索——这是数据科学家交付给信用风险团队的标准工作流 |
| **Validation Rigour**<br>验证严谨性 | EN: Fixed random seeds globally, documented temporal pooling vs. out-of-time validation trade-off, applied 3-fold cross-validation throughout<br>中文：全局锁定随机种子并记录完整清单；文档化"时序合并 vs 时间外推验证"的取舍论证；全程使用3折交叉验证 |

### Technical Stack · 技术栈

`Python` · `pandas` · `GeoPandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `shapely` · `NumPy` · `SciPy`

---

## Results · 模型结果

### Model Performance · 模型性能（对数房价预测）

| Model · 模型 | RMSE | R² | Strategy · 策略 |
|---|---|---|---|
| LinearRegression (baseline · 基线) | 0.3734 | 0.5624 | — |
| DecisionTree depth=3 (baseline · 基线) | 0.4655 | 0.3199 | — |
| **RandomForest + GridSearchCV** ★ | **0.3338** | **0.6503** | Bagging · 装袋法 |
| XGBoost + GridSearchCV | 0.3401 | 0.6371 | Boosting · 提升法 |

### Key Business Insight · 核心商业洞察

**EN:** The relationship between erosion proximity and property price is **non-linear with threshold effects** — prices don't drop gradually as you approach the coast. Instead, there's a measurable discount zone where the market suddenly prices in environmental risk. A linear model cannot capture this; ensemble tree methods can.

Bagging (Random Forest) outperformed Boosting (XGBoost) because coastal property data has inherently high variance from macro-level confounders (interest rates, local demand shocks). Bagging's variance-reduction mechanism is better suited to this signal-to-noise environment.

---

**中文：** 侵蚀距离与房价之间的关系是**非线性的，且存在阈值效应**——房价并非随靠近海岸而线性下跌，而是在某个临界区域内，市场会突然对环境风险进行大幅折价。线性模型无法捕捉这种突变模式，而集成树方法可以。

装袋法（Random Forest）优于提升法（XGBoost）的原因在于：海岸房产数据天然具有高方差——利率波动、本地需求冲击等宏观混杂因素使个别交易噪声极大。装袋法的方差缩减机制更适合这种低信噪比场景。

---

## Project Structure · 项目结构

```
├── Coastal_Property_Climate_Risk_Pricing_IB9JV.ipynb  ← 主分析 Notebook（双语标注）
├── requirements.txt                                    ← Python 依赖清单
├── README.md                                           ← 本文件
├── figures/                                            ← EDA 可视化输出（5张PNG）
│   ├── fig1_geo_scatter.png         地理散点分布图
│   ├── fig2_dist_logprice.png       对数房价分布图
│   ├── fig3_distributions.png       多变量分布面板图
│   ├── fig4_log_transform.png       对数变换前后对比图
│   └── fig5_dist_by_type.png        房产类型分组分布图
├── processed/                                          ← 预计算空间连接结果（43 MB）
│   ├── coastal_2020_clean.gpkg      2020年清洁数据
│   └── coastal_2025_clean.gpkg      2025年清洁数据
└── Exe_Summary test 4.15.2.pdf                         ← 执行摘要
```

---

## Reproducibility · 复现指南

### Quick Start · 快速开始

```bash
# 1. Clone and install dependencies · 克隆仓库并安装依赖
git clone https://github.com/XilaiWang/coastal-property-climate-risk.git
cd coastal-property-climate-risk
pip install -r requirements.txt

# 2. Download raw data (excluded due to GitHub 100 MB file limit)
#    下载原始数据（因 GitHub 单文件100 MB限制，未包含在仓库中）
#    HM Land Registry Price Paid Data (2020, 2025):
#    https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
#    将两个 CSV 文件放入项目根目录即可

# 3. Open the notebook · 打开 Notebook
jupyter notebook Coastal_Property_Climate_Risk_Pricing_IB9JV.ipynb
```

**EN:** The notebook auto-installs dependencies in its first cell and reads from pre-computed intermediate files (`processed/*.gpkg`) to ensure full reproducibility without raw data dependency.

**中文：** Notebook 的第一个代码单元会自动安装全部依赖包，并从预计算的中间文件（`processed/*.gpkg`）读取数据，确保即使没有原始数据文件也能完整复现所有分析结果。

### Data Sources · 数据来源

| Dataset · 数据集 | Provider · 提供方 | Purpose · 用途 |
|---|---|---|
| Price Paid Data (PPD)<br>房产成交价数据 | HM Land Registry<br>英国土地注册处 | EN: ~1.7M property transactions with sale prices<br>中文：约170万条含成交价的房产交易记录 |
| National Statistics Postcode Lookup (NSPL)<br>全国邮编坐标查询表 | ONS / Ordnance Survey<br>英国国家统计局/地形测量局 | EN: Postcode → coordinate mapping<br>中文：邮编 → 经纬度坐标映射 |
| National Coastal Erosion Risk Mapping (NCERM)<br>全国海岸侵蚀风险地图 | Defra / Environment Agency<br>英国环境食品与乡村事务部/环境署 | EN: Predicted erosion zones to 2100<br>中文：预测至2100年的海岸侵蚀边界线 |

---

## Project Context · 项目背景

| | |
|---|---|
| **Course · 课程** | IB9JV Programming for Data Analytics, Warwick Business School (2025–2026)<br>华威大学商学院《数据分析编程》 |
| **Scenario · 场景** | Risk management analyst at a British commercial bank<br>英国商业银行风险管理分析师 |
| **Assessment · 考核** | Full decision-record trail documenting every analytical choice with statistical justification<br>完整决策记录链——对每个分析选择给出统计学正当性论证 |
| **Award · 成绩** | MSc Business Analytics, University of Warwick<br>华威大学商业分析硕士 |

---

*Built by **Xilai Wang** · MSc Business Analytics @ Warwick Business School*  
*Interested in data science roles where rigorous methodology meets real business risk.*  
*寻求能将严谨分析方法论与真实商业风险相结合的數據科學岗位。*
