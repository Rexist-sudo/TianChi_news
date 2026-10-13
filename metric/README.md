# 公共数据预处理与评估

`data.py` 负责生成并读取固定的本地离线数据；`evaluation.py` 只读取模型预测和固定标签，计算 Recall@K、MRR@K。

## 预处理-data.py

`prepare_evaluation(train_clicks, *, n_users=50_000, random_state=42, cache_dir=...)` ：

- `fit_clicks`：训练集除验证标签外的全部点击，包括验证用户的剩余历史。
- `query_clicks`：验证用户在 `fit_clicks` 中的全部历史，用于生成推荐。
- `targets`：验证用户各自最后一次点击，每用户一条。

三张表均包含 `user_id`、`click_article_id`、`click_timestamp` 三列。首先对这三列完全相同的点击事件去重，再从至少有两次点击的训练用户中，用固定随机种子无放回抽取 50,000 人。点击时间相同则按原始行序确定末次事件；未抽中的用户保留全部点击。验证用户数不足时直接报错，不自动缩减规模。

【注意】：其他用户的训练点击可能晚于某个验证用户的标签时间，因此会发生**时间穿越**(即未来信息拟合过去点击)

## 在任意 Task 中使用

```python
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))

from metric.data import prepare_evaluation
from metric.evaluation import (
    evaluate_predictions,
    evaluate_csv,
    save_predictions
)

eval_dir = sys.path / "cache" / "offline_eval"
train_clicks = pd.read_csv(project_root / "data" / "train_click_log.csv")
fit_clicks, query_clicks, targets = prepare_evaluation(
    train_clicks, cache_dir=eval_dir,
)

# 在这里用 fit_clicks 拟合模型，为 query_clicks 的用户生成 predictions_df。
# 各模型可以过滤、构建自身的训练样本，随后统计实际参与拟合的去重用户数。
train_users = fit_clicks["user_id"].nunique()  # 适用于直接使用全部 fit 的 ItemCF
```

完成预测后保存 CSV，再调用公共评估模块：

```python
result_dir = project_root / "Task-1" / "results" / "offline_eval"
prediction_path = save_predictions(predictions_df, result_dir / "top-50.csv")
scores = evaluate_csv(
    prediction_path,
    eval_dir / "targets.csv",
    train_users=train_users,
    output_path=result_dir / "metrics.json",
)
```

预测支持长表（`user_id, click_article_id`，附 `recall_rank`、`rank` 或 `pred_score`）或 Top-K 宽表（`user_id, article_1, ..., article_K`）。宽表不足 K 篇时尾部留空。Top50 的默认指标为 `R@5`、`R@50`、`MRR@5`、`MRR@50`，另记录 `train_users` 与 `validation_users`。

所有指标以固定标签用户数为分母；有标签但无预测的用户计 0 分，标签之外的预测用户忽略。

不能通过过滤候选或查询用户改变评估分母。
