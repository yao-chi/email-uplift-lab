# 数据说明

## 文件

`hillstrom_email.csv` —— 64,000 行 × 12 列，UTF-8，逗号分隔。

## 来源

Kevin Hillstrom, *MineThatData E-Mail Analytics And Data Mining Challenge*（2008）。
原始数据由 MineThatData 公开发布用于数据挖掘教学与分析竞赛，长期被 Kaggle、`scikit-uplift` / `causalml`
等因果推断库作为 uplift 建模的标准数据集使用。

本仓库保留的副本与原始文件逐字节一致：

| 项 | 值 |
|---|---|
| 文件大小 | 3,964,977 字节 |
| SHA256 | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` |
| 行数（不含表头） | 64,000 |

`scripts/run_all.py` 启动时会重新计算 SHA256 并写进 `reports/results.json`，可用来自证数据未被改动。

## 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `recency` | int | 距上次购买的天数分段（1 = 最近，12 = 最久） |
| `history_segment` | str | 过去 12 个月消费额区间（7 档） |
| `history` | float | 过去 12 个月消费额（美元） |
| `mens` | 0/1 | 是否买过男装 |
| `womens` | 0/1 | 是否买过女装 |
| `zip_code` | str | 城乡类型：Surburban / Rural / Urban（原字段拼写即为 Surburban） |
| `newbie` | 0/1 | 是否新客 |
| `channel` | str | 购买渠道：Phone / Web / Multichannel |
| `segment` | str | **实验分组**：Mens E-Mail / Womens E-Mail / No E-Mail |
| `visit` | 0/1 | 实验后两周内是否访问网站 |
| `conversion` | 0/1 | 是否转化 |
| `spend` | float | 消费额（美元，未转化为 0） |

## 使用注意

1. **没有用户 ID**。文件中有 6,562 行完全重复的记录（10.3%），这不是脏数据，而是不同用户恰好
   12 个字段取值相同。按行去重会人为缩小样本并抬高基准率，因此本项目**不做去重**。
2. `spend` 零膨胀（99.1% 为 0），均值类指标方差大，任何 ARPU/人均消费结论都要带置信区间。
3. 处理变量是 `segment`，不是 `visit`/`conversion`——后者是结果，不能进特征。

## 许可

原始数据由 MineThatData 公开发布用于分析用途。本仓库在此引用并保留原始文件以便复现；
如用于商业用途，请自行确认原始发布方的条款。代码部分为 MIT（见根目录 `LICENSE`）。