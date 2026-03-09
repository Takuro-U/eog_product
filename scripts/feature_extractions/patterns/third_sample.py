"""
EOG 4ch 特徴量セット（6次元）

チャンネル構成:
    - ch1: 横方向位置成分（水平 EOG・低周波）
    - ch3: 縦方向位置成分（垂直 EOG・低周波）
    ※ ch2, ch4（サッカード成分）は本セットでは不使用

対象クラス: 「上」「下」「左」「右」「中央」「移動中」

特徴量カテゴリ:
    1. 位置レベル・分布 (6次元): 保持位置判別の主軸

設計方針:
    保持位置（上下左右中央）の識別精度を最大化する構成。
    移動中クラスの識別精度低下は許容している。

second_sample.py（14次元）からの除外内容:
    - slope_h, slope_v           : 移動中クラス特化、保持判別への寄与なし
    - rms_sac_h, rms_sac_v       : 保持5クラス間での識別力なし
    - n_sac_h, n_sac_v           : 同上
    - sac_energy_ratio_h/v       : サッカード活動度除外に伴い意味を失う
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import (
    BaseRealtimeExtractor,
    create_windows,
    determine_label,
    extract_channels,
)


# ============================================================
# 特徴量定義（6次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 2.1 位置レベル・分布 (6)
    "mean_h",
    "mean_v",
    "median_h",
    "median_v",
    "std_h",
    "std_v",
]


# ============================================================
# パラメータ
# ============================================================

# ゼロ除算防止用の小さな値
EPSILON = 1e-8


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(channels: dict[str, np.ndarray]) -> dict[str, float]:
    """
    チャンネルデータから6次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
                  ※ ch2, ch4 は本関数内では参照しない

    Returns:
        特徴量辞書（6次元）
    """
    ch1 = channels["ch1"]  # 横方向位置
    ch3 = channels["ch3"]  # 縦方向位置

    features = {}

    # ---------------------------------------------------------
    # 2.1 位置レベル・分布 (6次元)
    # ---------------------------------------------------------
    features["mean_h"] = float(np.mean(ch1))
    features["mean_v"] = float(np.mean(ch3))
    features["median_h"] = float(np.median(ch1))
    features["median_v"] = float(np.median(ch3))
    features["std_h"] = float(np.std(ch1))
    features["std_v"] = float(np.std(ch3))

    return features


# ============================================================
# 単一窓の特徴量抽出
# ============================================================

def extract_features(
    samples: list[Any],
    *,
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
) -> dict[str, float]:
    """
    窓内のサンプルから特徴量を抽出

    Args:
        samples: 窓サイズ分のサンプルリスト
        channel_keys: チャンネルのキー名

    Returns:
        特徴量辞書（6次元）

    Raises:
        ValueError: サンプルが空の場合
    """
    if not samples:
        raise ValueError("samples must not be empty")

    channels = extract_channels(samples, channel_keys)
    return _compute_features(channels)


# ============================================================
# バッチ処理
# ============================================================

def batch_extract_from_dataframe(
    df: pd.DataFrame,
    window_size: int,
    stride: int | None = None,
    *,
    channel_columns: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    label_column: str | None = None,
    label_strategy: str = "center",
) -> pd.DataFrame:
    """
    DataFrameから特徴量を一括抽出

    Args:
        df: 入力DataFrame（各行が1サンプル）
        window_size: 窓サイズ
        stride: ストライド（Noneの場合はwindow_size）
        channel_columns: チャンネル列名
        label_column: ラベル列名（Noneならラベルなし）
        label_strategy: ラベル決定戦略

    Returns:
        特徴量DataFrame（6次元 + label列）
    """
    if stride is None:
        stride = window_size

    results = []

    for start, window_df in create_windows(df, window_size, stride):
        # チャンネルデータ抽出
        channels = {
            col: window_df[col].values.astype(np.float64)
            for col in channel_columns
        }

        # 特徴量計算
        features = _compute_features(channels)

        # ラベル付与（あれば）
        if label_column is not None and label_column in df.columns:
            features["label"] = determine_label(
                window_df[label_column], label_strategy
            )

        results.append(features)

    return pd.DataFrame(results)


def batch_extract_from_csv(
    csv_path: str | Path,
    window_size: int,
    stride: int | None = None,
    *,
    channel_columns: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    label_column: str | None = "label",
    label_strategy: str = "center",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    CSVファイルから特徴量を一括抽出

    Args:
        csv_path: 入力CSVファイルパス
        window_size: 窓サイズ
        stride: ストライド
        channel_columns: チャンネル列名
        label_column: ラベル列名（存在しなければ無視）
        label_strategy: ラベル決定戦略
        output_path: 出力CSVパス（Noneなら保存しない）

    Returns:
        特徴量DataFrame
    """
    df = pd.read_csv(csv_path)

    # label_columnが存在しない場合はNone扱い
    actual_label_column = label_column
    if label_column is not None and label_column not in df.columns:
        actual_label_column = None

    features_df = batch_extract_from_dataframe(
        df,
        window_size=window_size,
        stride=stride,
        channel_columns=channel_columns,
        label_column=actual_label_column,
        label_strategy=label_strategy,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)

    return features_df


# ============================================================
# リアルタイム処理
# ============================================================

class RealtimeExtractor(BaseRealtimeExtractor):
    """
    リアルタイム特徴量抽出器（6次元）

    Example:
        >>> extractor = RealtimeExtractor(window_size=125, stride=62)
        >>> for sample in stream:
        ...     features = extractor.push(sample)
        ...     if features is not None:
        ...         prediction = model.predict([features])
    """

    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """特徴量計算（モジュールレベル関数を呼び出し）"""
        return _compute_features(channels)


# ============================================================
# ユーティリティ
# ============================================================

def get_feature_columns() -> list[str]:
    """特徴量カラム名のリストを取得（6次元）"""
    return FEATURE_COLUMNS.copy()