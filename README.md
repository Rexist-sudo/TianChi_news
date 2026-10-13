# TianChi_news

阿里云日常赛：零基础入门推荐系统 - 新闻推荐，赛题以预测用户未来点击新闻文章为任务，该数据来自某新闻APP平台的用户交互数据，包括30万用户，近300万次点击，共36万多篇不同的新闻文章，同时每篇新闻文章有对应的embedding向量表示。

<p align="left">
  <a href="https://tianchi.aliyun.com/competition/entrance/531842/information">
    <img src="https://img.shields.io/badge/Tianchi-新闻推荐-ff6a00?style=flat-square" alt="Tianchi">
  </a>
  <img src="https://img.shields.io/badge/Task-推荐系统-4e7ad0?style=flat-square" alt="Task">
  <img src="https://img.shields.io/badge/MRR-Mean%20Reciprocal%20Rank-3aa675?style=flat-square" alt="Metric">
</p>

## 数据说明

### 数据划分

| 数据集   | 用户数 | 用途                   |
| -------- | ------ | ---------------------- |
| 训练集   | 20 万  | 模型训练               |
| 测试集 A | 5 万   | 公开测试、生成线上提交 |
| 测试集 B | 5 万   | 最终评测（本地不包含） |

### 数据文件

| 文件名                | 说明                    |
| --------------------- | ----------------------- |
| `train_click_log.csv` | 训练集用户点击日志      |
| `testA_click_log.csv` | 测试集用户点击日志      |
| `articles.csv`        | 新闻文章信息表          |
| `articles_emb.csv`    | 新闻文章 Embedding 向量 |
| `sample_submit.csv`   | 提交样例                |

### 字段说明

| 字段                  | 说明                                |
| --------------------- | ----------------------------------- |
| `user_id`             | 用户 ID                             |
| `click_article_id`    | 点击文章 ID                         |
| `click_timestamp`     | 点击时间戳                          |
| `click_environment`   | 点击环境                            |
| `click_deviceGroup`   | 点击设备组                          |
| `click_os`            | 点击操作系统                        |
| `click_country`       | 点击城市                            |
| `click_region`        | 点击地区                            |
| `click_referrer_type` | 点击来源类型                        |
| `article_id`          | 文章 ID，与 `click_article_id` 对应 |
| `category_id`         | 文章类型 ID                         |
| `created_at_ts`       | 文章创建时间戳                      |
| `words_count`         | 文章字数                            |
| `emb_1 ~ emb_249`     | 文章 Embedding 向量（250 维）       |

---

## 评估指标 — MRR

**MRR（Mean Reciprocal Rank）**：对每个测试用户，若模型推荐的 Top-5 文章命中用户最后一次真实点击，则按排名计分。

单个用户得分：

$$
\text{score}(user)
=
\sum_{k=1}^{5} \frac{s(user,k)}{k},
\quad
s(user,k)
=
\begin{cases}
1 & \text{第 } k \text{ 位命中} \\
0 & \text{未命中}
\end{cases}
$$
最终得分 = 所有用户得分的平均值。

---

## 提交格式

提交文件为 CSV 格式，每行包含用户 ID 及预测的 Top-5 文章 ID（按概率降序排列）：

```csv
user_id,article_1,article_2,article_3,article_4,article_5
200000,1,2,3,4,5
200001,1,2,3,4,5
200002,1,2,3,4,5
200003,1,2,3,4,5
```

> 提交前请确保格式与 `sample_submit.csv` 一致，文件后缀名为 `.csv`。

---
## 本地部署
本地环境：

```cmd
python -m venv .venv
python -m pip install -r requirements.txt
```


检查环境是否启用：

```cmd
python -c "import sys; print(sys.executable)"
```



## 本地离线评估

本地离线评估使用固定的 **train末次点击划分**：从至少有两次点击的训练用户中，按固定随机种子 `42` 无放回抽取 **50,000 个验证用户**，留出各用户最后一次点击作为标签，其余点击组成公共拟合数据。验证用户的历史仍参与模型拟合，未抽中的训练用户保留全部点击；TestA 不参与本地离线划分或拟合。

公共模块位于 [`metric/`](metric/README.md)，职责如下：

- **数据预处理模块 `metric/data.py`**：`prepare_evaluation(train_clicks, ...)` 返回 `fit_clicks, query_clicks, targets`。首次运行将三份数据保存到项目根目录 `cache/offline_eval/`；后续运行直接读取，不重新抽样。各 Task 复用同一份数据，模型可按自身需要构建训练样本。
- **评估模块 `metric/evaluation.py`**：只读取模型预测和固定标签计算 Recall@K、MRR@K，不负责抽样、数据划分或训练样本构建。

| 缓存文件 | 内容 |
| --- | --- |
| `cache/offline_eval/fit_clicks.csv` | 全部训练点击去掉验证用户的末次点击，用于模型拟合 |
| `cache/offline_eval/query_clicks.csv` | 验证用户去掉末次点击后的历史，用于生成预测；是 `fit_clicks` 的子集 |
| `cache/offline_eval/targets.csv` | 50,000 个验证用户各自的末次点击，每用户一个标签 |

点击事件先按 `user_id, click_article_id, click_timestamp` 去重，再按时间排序；时间相同时按原始行序确定末次点击。可用用户不足 50,000 人或缓存缺失部分文件时会报错，不会自动缩减规模或重新划分。缓存已存在时以缓存为准，修改种子或用户数不会覆盖已有划分。

在项目根目录运行的示例：

```python
import pandas as pd
from metric.data import prepare_evaluation
from metric.evaluation import evaluate_csv

train_clicks = pd.read_csv("data/train_click_log.csv")
fit_clicks, query_clicks, targets = prepare_evaluation(train_clicks)
# 模型基于 fit_clicks 拟合，并为 query_clicks 中的用户生成预测 CSV。

scores = evaluate_csv(
    "Task-1/results/offline_eval/top-50.csv", # 以 Task-1为例
    "cache/offline_eval/targets.csv",
    train_users=fit_clicks["user_id"].nunique(),  # 实际参与拟合的数据
    output_path="Task-1/results/offline_eval/metrics.json",
)
```

- 导入方式及完整数据划分示例见 [metric 使用说明](metric/README.md)。

- 离线输出位于 `Task-1/results/offline_eval/`，包含 `top-50.csv` 和 `metrics.json`。JSON 保留 `R@5`、`R@50`、`MRR@5`、`MRR@50`，以及处理后实际参与拟合的去重用户数 `train_users` 和实际参与计分的用户数 `validation_users`。`train_users` 必须由模型侧统计传入；后续模型应按最终有效训练样本统计。公共划分保存在根目录 `cache/offline_eval/`，模型自身的相似度等缓存仍保存在 `Task-1/cache/offline_eval/`。旧的 Task-1 私有 `targets.csv` 不再使用。`offline=False` 则用全部 train + TestA 可见点击拟合，为 TestA 用户生成 Top5 提交格式 CSV，保存到 `Task-1/results/online/`。

- `Task-1/baseline.ipynb` 的 `offline=True` 使用上述公共划分：ItemCF 相似度与热门统计仅基于 `fit_clicks`，仅为 `query_clicks` 中的 50,000 个用户生成 Top50 CSV。有标签但零候选的用户仍计入分母，记 0 分。



## 进度

### Task-1

https://tianchi.aliyun.com/notebook/144449


### Task-2

https://tianchi.aliyun.com/notebook/144451


### Task-3

https://tianchi.aliyun.com/notebook/144452


### Task-4

https://tianchi.aliyun.com/notebook/144453

### Task-5

https://tianchi.aliyun.com/notebook/144454



### TODO

1. []总结：Task内容、项目结果
   
2. []新增指标：NDCG@k、排序AUC
  

