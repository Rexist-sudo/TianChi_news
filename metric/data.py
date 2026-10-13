"""复用公共 train 留一划分；模型拟合前先移除验证标签。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CLICK_COLUMNS = ["user_id", "click_article_id", "click_timestamp"]
OFFLINE_EVAL_DIR = Path(__file__).resolve().parents[1] / "cache" / "offline_eval"


def _ordered_clicks(clicks: pd.DataFrame) -> pd.DataFrame:
    missing = set(CLICK_COLUMNS) - set(clicks.columns)
    if missing:
        raise ValueError(f"Missing click columns: {sorted(missing)}")
    result = clicks[CLICK_COLUMNS].copy()
    for column in CLICK_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain non-null integers")
        result[column] = values.astype("int64")
    # 去掉完全相同的点击事件。
    return (
        result.drop_duplicates(CLICK_COLUMNS)
        .sort_values(["user_id", "click_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def split_last_click(
    clicks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多击用户留出末次点击；单击用户保留历史，不生成标签。

    返回 ``(history, targets)``，两张表均为 CLICK_COLUMNS 三列。相同时间戳
    按原始行序确定最后一条。标签中的 click_timestamp 用于追溯留出事件，
    指标只使用 user_id 和 click_article_id。
    """
    ordered = _ordered_clicks(clicks)
    sizes = ordered.groupby("user_id")["user_id"].transform("size")
    last = ~ordered["user_id"].duplicated(keep="last")
    held_out = sizes.ge(2) & last
    return (
        ordered.loc[~held_out].reset_index(drop=True),
        ordered.loc[held_out].reset_index(drop=True),
    )


def prepare_evaluation(
    train_clicks: pd.DataFrame,
    *,
    n_users: int = 50_000,
    random_state: int = 42,
    cache_dir: str | Path = OFFLINE_EVAL_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回并缓存固定的 ``(fit_clicks, query_clicks, targets)``。

    首次运行从去重后至少有两次点击的训练用户中，以固定种子无放回抽取
    n_users 人，各留出最后一次点击作为 targets。fit_clicks 包含训练集
    除这些标签外的全部点击；query_clicks 是验证用户在 fit_clicks 中的历史。
    三张表均为 CLICK_COLUMNS 三列，各模型可基于公共 fit 自行构建训练样本。

    缓存位于项目根目录 cache/offline_eval 

    注意：不保证其他用户的事件早于每一个 query 的预测时刻，因此会发生时间穿越
    """
    cache_dir = Path(cache_dir)
    paths = [cache_dir / f"{name}.csv" for name in ("fit_clicks", "query_clicks", "targets")]
    present = [path.exists() for path in paths]
    if all(present):
        fit, query, targets = (
            pd.read_csv(path, usecols=CLICK_COLUMNS, dtype={column: "int64" for column in CLICK_COLUMNS})
            for path in paths
        )
        return fit, query, targets
    if any(present):
        missing = [path.name for path in paths if not path.exists()]
        raise FileNotFoundError(
            f"Incomplete offline evaluation cache in {cache_dir}: missing {missing}. "
            "Restore the complete split or use a new cache_dir; no resampling was performed."
        )
    if isinstance(n_users, bool) or not isinstance(n_users, int) or n_users <= 0:
        raise ValueError("n_users must be a positive integer")

    train = _ordered_clicks(train_clicks)
    sizes = train.groupby("user_id").size()
    eligible_users = sizes.index[sizes.ge(2)].to_series()
    if len(eligible_users) < n_users:
        raise ValueError(
            f"Requested {n_users} validation users, but only {len(eligible_users)} "
            "training users have at least two distinct clicks."
        )
    validation_users = eligible_users.sample(n=n_users, random_state=random_state)
    # 抽样验证集
    is_validation = train["user_id"].isin(validation_users)
    held_out = is_validation & ~train["user_id"].duplicated(keep="last")
    # 取验证集用户的最后一条点击记录
    fit = train.loc[~held_out].reset_index(drop=True)
    query = train.loc[is_validation & ~held_out].reset_index(drop=True)
    targets = train.loc[held_out].reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in zip((fit, query, targets), paths):
        frame.to_csv(path, index=False)
    return fit, query, targets
