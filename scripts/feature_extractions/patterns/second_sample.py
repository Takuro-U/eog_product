"""
EOG 4ch 特徴量セット（14次元）

チャンネル構成:
    - ch1: 横方向位置成分（水平 EOG・低周波）
    - ch2: 横方向サッカード成分（水平サッカード時スパイク）
    - ch3: 縦方向位置成分（垂直 EOG・低周波）
    - ch4: 縦方向サッカード成分（垂直サッカード時スパイク）

対象クラス: 「上」「下」「左」「右」「中央」「移動中」

特徴量カテゴリ:
    1. 位置レベル・分布 (6次元): 方向・中央判定の主軸
    2. 位置トレンド・変化量 (2次元): 移動中の検出
    3. サッカード活動度 (4次元): 移動中の主軸
    4. 統合・関係特徴 (2次元): サッカード方向性

first_sample.py（29次元）から以下を除外:
    - range_h, range_v    : std との冗長性
    - delta_h, delta_v    : slope との高相関
    - abs_mean_sac_h/v    : rms との冗長性
    - gaze_radius, gaze_angle, sac_energy_h/v/total : mean_h/v からの派生または冗長
    - energy_sac_*_first/last : 保持位置判別への寄与が限定的
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
import pandas as pd

from .. import (
    BaseRealtimeExtractor,
    create_windows,
    determine_label,
    extract_channels,
)


# ============================================================
# 特徴量定義（14次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 2.1 位置レベル・分布 (6)
    "mean_h",
    "mean_v",
    "median_h",
    "median_v",
    "std_h",
    "std_v",
    # 2.2 位置トレンド・変化量 (2)
    "slope_h",
    "slope_v",
    # 2.3 サッカード活動度 (4)
    "rms_sac_h",
    "rms_sac_v",
    "n_sac_h",
    "n_sac_v",
    # 2.4 統合・関係特徴 (2)
    "sac_energy_ratio_h",
    "sac_energy_ratio_v",
]


# ============================================================
# パラメータ
# ============================================================

# サッカードピーク検出の閾値（標準偏差の倍数）
SACCADE_PEAK_THRESHOLD_STD = 2.0

# ゼロ除算防止用の小さな値
EPSILON = 1e-8


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(channels: dict[str, np.ndarray]) -> dict[str, float]:
    """
    チャンネルデータから14次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}

    Returns:
        特徴量辞書（14次元）
    """
    ch1 = channels["ch1"]  # 横方向位置
    ch2 = channels["ch2"]  # 横方向サッカード
    ch3 = channels["ch3"]  # 縦方向位置
    ch4 = channels["ch4"]  # 縦方向サッカード

    n_samples = len(ch1)
    t = np.arange(n_samples)

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

    # ---------------------------------------------------------
    # 2.2 位置トレンド・変化量 (2次元)
    # ---------------------------------------------------------
    features["slope_h"] = _compute_slope(t, ch1)
    features["slope_v"] = _compute_slope(t, ch3)

    # ---------------------------------------------------------
    # 2.3 サッカード活動度 (4次元)
    # ---------------------------------------------------------
    rms_sac_h = float(np.sqrt(np.mean(ch2 ** 2)))
    rms_sac_v = float(np.sqrt(np.mean(ch4 ** 2)))

    features["rms_sac_h"] = rms_sac_h
    features["rms_sac_v"] = rms_sac_v
    features["n_sac_h"] = float(_count_peaks(ch2))
    features["n_sac_v"] = float(_count_peaks(ch4))

    # ---------------------------------------------------------
    # 2.4 統合・関係特徴 (2次元)
    # ---------------------------------------------------------
    sac_energy_total = rms_sac_h + rms_sac_v + EPSILON

    features["sac_energy_ratio_h"] = rms_sac_h / sac_energy_total
    features["sac_energy_ratio_v"] = rms_sac_v / sac_energy_total

    return features


def _compute_slope(t: np.ndarray, y: np.ndarray) -> float:
    """
    単回帰の傾きを計算: y = a*t + b の a

    最小二乗法: a = Σ((t - t̄)(y - ȳ)) / Σ((t - t̄)²)
    """
    t_mean = np.mean(t)
    y_mean = np.mean(y)

    numerator = np.sum((t - t_mean) * (y - y_mean))
    denominator = np.sum((t - t_mean) ** 2)

    if denominator < EPSILON:
        return 0.0

    return float(numerator / denominator)


def _count_peaks(data: np.ndarray) -> int:
    """
    閾値以上のピーク数をカウント

    閾値: データの標準偏差 × SACCADE_PEAK_THRESHOLD_STD
    """
    threshold = np.std(data) * SACCADE_PEAK_THRESHOLD_STD

    # 絶対値でピーク検出（正負両方のスパイクを検出）
    abs_data = np.abs(data)

    # scipy.signal.find_peaks で局所最大値を検出
    peaks, _ = signal.find_peaks(abs_data, height=threshold)

    return len(peaks)


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
        特徴量辞書（14次元）

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
        特徴量DataFrame（14次元 + label列）
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
    リアルタイム特徴量抽出器（14次元）

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
    """特徴量カラム名のリストを取得（14次元）"""
    return FEATURE_COLUMNS.copy()