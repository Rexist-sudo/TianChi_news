"""使用固定标签分母评估召回长表或 Top-K 宽表，不依赖模型及 Task 路径。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _integers(values: pd.Series, name: str, *, positive: bool = False) -> pd.Series:
    numbers = pd.to_numeric(values, errors="raise")
    if numbers.isna().any() or not np.isfinite(numbers).all() or (numbers % 1 != 0).any():
        raise ValueError(f"{name} must contain non-null integers")
    if positive and numbers.le(0).any():
        raise ValueError(f"{name} must be positive")
    return numbers.astype("int64")


def _targets_frame(targets: pd.DataFrame | Mapping[int, int]) -> pd.DataFrame:
    if isinstance(targets, Mapping):
        targets = pd.DataFrame(list(targets.items()), columns=["user_id", "click_article_id"])
    if not {"user_id", "click_article_id"}.issubset(targets.columns):
        raise ValueError("Targets require user_id and click_article_id columns")
    frame = targets[["user_id", "click_article_id"]].copy()
    if frame.empty:
        raise ValueError("No held-out targets: metrics would have a zero denominator")
    for column in frame.columns:
        frame[column] = _integers(frame[column], column)
    if frame["user_id"].duplicated().any():
        raise ValueError("Targets must contain exactly one article per user")
    return frame


def _ranked_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """统一为 user_id/article/rank，宽表额外返回其保存的排名上限。"""
    if "user_id" not in predictions:
        raise ValueError("Predictions require a user_id column")
    raw = predictions.copy()
    raw["user_id"] = _integers(raw["user_id"], "user_id")
    article_columns = sorted(
        ((int(match.group(1)), column) for column in raw.columns
         if (match := re.fullmatch(r"article_([1-9]\d*)", str(column)))),
        key=lambda pair: pair[0],
    )
    capacity = None
    if article_columns:
        capacity = len(article_columns)
        if [rank for rank, _ in article_columns] != list(range(1, capacity + 1)):
            raise ValueError("Wide CSV columns must be consecutive article_1 ... article_K")
        if raw["user_id"].duplicated().any():
            raise ValueError("Wide predictions must have one row per user")
        pieces = []
        for rank, column in article_columns:
            piece = raw.loc[raw[column].notna(), ["user_id", column]].rename(
                columns={column: "click_article_id"}
            )
            piece["rank"] = rank
            pieces.append(piece)
        frame = pd.concat(pieces, ignore_index=True)
    else:
        if "click_article_id" not in raw:
            raise ValueError("Use article_1 ... article_K, or long predictions with click_article_id")
        frame = raw
        rank_column = next((name for name in ("recall_rank", "rank") if name in frame), None)
        if rank_column:
            frame["rank"] = _integers(frame[rank_column], rank_column, positive=True)
        elif "pred_score" in frame:
            frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="raise")
            if not np.isfinite(frame["pred_score"]).all():
                raise ValueError("pred_score must contain finite numbers")
            frame["click_article_id"] = _integers(frame["click_article_id"], "click_article_id")
            frame = frame.sort_values(
                ["user_id", "pred_score", "click_article_id"],
                ascending=[True, False, True], kind="mergesort",
            )
            frame["rank"] = frame.groupby("user_id").cumcount() + 1
        else:
            raise ValueError("Long predictions require recall_rank, rank, or pred_score")
    frame = frame[["user_id", "click_article_id", "rank"]].copy()
    frame["click_article_id"] = _integers(frame["click_article_id"], "click_article_id")
    frame["rank"] = _integers(frame["rank"], "rank", positive=True)
    if frame.duplicated(["user_id", "click_article_id"]).any():
        raise ValueError("Duplicate user/article predictions are not allowed")
    if frame.duplicated(["user_id", "rank"]).any():
        raise ValueError("Duplicate ranks within a user are not allowed")
    frame = frame.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)
    expected_rank = frame.groupby("user_id").cumcount() + 1
    if not frame["rank"].eq(expected_rank).all():
        raise ValueError("Ranks must start at 1 without gaps; wide CSV blanks must be trailing")
    return frame, capacity


def evaluate_predictions(
    predictions: pd.DataFrame,
    targets: pd.DataFrame | Mapping[int, int],
    ks: Sequence[int] | None = None,
    *,
    train_users: int,
) -> dict[str, int | float]:
    """计算 Recall@K 和 MRR@K，每用户一个标签，零候选用户计 0 分。

    长表按 recall_rank/rank（优先）或 pred_score 排序；同分按文章 ID 升序。
    宽表使用 article_1 ... article_K 的位置，不足 K 篇可用尾部空值。
  
    默认结果仅包含 R@5、R@50、MRR@5、MRR@50 和两个用户数。

    train_users 必须由模型侧传入：过滤、构建训练样本后实际参与拟合的去重用户数。
。
    validation_users 是有标签的用户数，也是所有指标的分母。
    标签之外的用户忽略；重复候选/重复标签用户会报错。
    """
    if isinstance(train_users, bool) or not isinstance(train_users, (int, np.integer)) or train_users < 0:
        raise ValueError("train_users must be a nonnegative integer counted from actual training data")
    labels = _targets_frame(targets)
    ranked, capacity = _ranked_predictions(predictions)
    if ks is None:
        ks = [k for k in (5, 50) if capacity is None or k <= capacity]
        if not ks:
            ks = [capacity] # type: ignore
    if not ks or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0 for k in ks):
        raise ValueError("ks must be a nonempty sequence of positive integers")
    cutoffs = sorted(set(int(k) for k in ks))
    if capacity is not None and max(cutoffs) > capacity:
        raise ValueError(f"CSV only stores Top{capacity}; cannot evaluate K={max(cutoffs)}")

    label_users = pd.Index(labels["user_id"])
    evaluated = ranked.loc[ranked["user_id"].isin(label_users)]
    hits = evaluated.merge(labels, on=["user_id", "click_article_id"], how="inner", validate="many_to_one")
    denominator = len(labels)
    result: dict[str, int | float] = {
        "train_users": int(train_users),
        "validation_users": denominator,
    }
    for k in cutoffs:
        result[f"R@{k}"] = float(hits["rank"].le(k).sum() / denominator)
    for k in cutoffs:
        hit_ranks = hits.loc[hits["rank"].le(k), "rank"]
        result[f"MRR@{k}"] = float((1.0 / hit_ranks).sum() / denominator)
    return result


def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    """保存预测 CSV，自动创建父目录；可传入召回长表或已排序的 Top-K 宽表。

    各模型在导出预测时调用本函数，随后将返回的路径传给 evaluate_csv。
    示例：save_predictions(predictions_df, result_dir / "top-50.csv")。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def evaluate_csv(
    predictions_path: str | Path,
    targets_path: str | Path,
    ks: Sequence[int] | None = None,
    output_path: str | Path | None = None,
    *,
    train_users: int,
) -> dict[str, int | float]:
    """评估已存在的 CSV；指定 output_path 时自动创建目录并保存 JSON。

    必须传入实际训练用户数。预测 CSV 可先通过 save_predictions 保存，
    使预测和指标的输出目录都由公共模块创建。
    """
    predictions_path, targets_path = Path(predictions_path), Path(targets_path)
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.resolve() in {predictions_path.resolve(), targets_path.resolve()}:
            raise ValueError("Metrics output must not overwrite input CSV files")
    # 显式的空候选以 CSV 空单元格表示；整数 ID 避免先转 float 丢失精度。
    prediction_header = pd.read_csv(predictions_path, nrows=0)
    id_columns = [column for column in prediction_header.columns
                  if column in {"user_id", "click_article_id"} or re.fullmatch(r"article_[1-9]\d*", column)]
    predictions = pd.read_csv(predictions_path, dtype={column: "Int64" for column in id_columns})
    targets = pd.read_csv(targets_path, dtype={"user_id": "Int64", "click_article_id": "Int64"})
    result = evaluate_predictions(predictions, targets, ks=ks, train_users=train_users)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result
