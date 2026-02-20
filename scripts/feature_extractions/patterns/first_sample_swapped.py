"""
EOG 4ch 特徴量セット（29次元） - チャンネル入れ替え版

チャンネル構成（first_sample から ch1↔ch2、ch3↔ch4 を入れ替え）:
    - ch1: 横方向サッカード成分（水平サッカード時スパイク）※元ch2
    - ch2: 横方向位置成分（水平 EOG・低周波）※元ch1
    - ch3: 縦方向サッカード成分（垂直サッカード時スパイク）※元ch4
    - ch4: 縦方向位置成分（垂直 EOG・低周波）※元ch3

対象クラス: 「上」「下」「左」「右」「中央」「移動中」

特徴量カテゴリ:
    1. 位置レベル・分布 (8次元): 方向・中央判定の主軸
    2. 位置トレンド・変化量 (4次元): 移動中の検出
    3. サッカード活動度 (6次元): 移動中の主軸
    4. 統合・関係特徴 (7次元): 方向とサッカード方向性
    5. サッカード時間局在 (4次元): 遷移タイミング
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
# 特徴量定義（29次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 2.1 位置レベル・分布 (8)
    "mean_h",
    "mean_v",
    "median_h",
    "median_v",
    "range_h",
    "range_v",
    "std_h",
    "std_v",
    # 2.2 位置トレンド・変化量 (4)
    "slope_h",
    "slope_v",
    "delta_h",
    "delta_v",
    # 2.3 サッカード活動度 (6)
    "abs_mean_sac_h",
    "abs_mean_sac_v",
    "rms_sac_h",
    "rms_sac_v",
    "n_sac_h",
    "n_sac_v",
    # 2.4 統合・関係特徴 (7)
    "gaze_radius",
    "gaze_angle",
    "sac_energy_h",
    "sac_energy_v",
    "sac_energy_total",
    "sac_energy_ratio_h",
    "sac_energy_ratio_v",
    # 2.5 サッカード時間局在 (4)
    "energy_sac_h_first",
    "energy_sac_h_last",
    "energy_sac_v_first",
    "energy_sac_v_last",
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
    チャンネルデータから29次元の特徴量を計算
    
    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
    
    Returns:
        特徴量辞書（29次元）
    """
    # チャンネル入れ替え: ch1↔ch2、ch3↔ch4
    ch1_orig = channels["ch2"]  # 横方向位置（元ch1）
    ch2_orig = channels["ch1"]  # 横方向サッカード（元ch2）
    ch3_orig = channels["ch4"]  # 縦方向位置（元ch3）
    ch4_orig = channels["ch3"]  # 縦方向サッカード（元ch4）
    
    n_samples = len(ch1_orig)
    t = np.arange(n_samples)
    half = n_samples // 2
    
    features = {}
    
    # ---------------------------------------------------------
    # 2.1 位置レベル・分布 (8次元)
    # ---------------------------------------------------------
    mean_h = float(np.mean(ch1_orig))
    mean_v = float(np.mean(ch3_orig))
    
    features["mean_h"] = mean_h
    features["mean_v"] = mean_v
    features["median_h"] = float(np.median(ch1_orig))
    features["median_v"] = float(np.median(ch3_orig))
    features["range_h"] = float(np.max(ch1_orig) - np.min(ch1_orig))
    features["range_v"] = float(np.max(ch3_orig) - np.min(ch3_orig))
    features["std_h"] = float(np.std(ch1_orig))
    features["std_v"] = float(np.std(ch3_orig))
    
    # ---------------------------------------------------------
    # 2.2 位置トレンド・変化量 (4次元)
    # ---------------------------------------------------------
    # 単回帰の傾き: y = a*t + b の a を計算
    features["slope_h"] = _compute_slope(t, ch1_orig)
    features["slope_v"] = _compute_slope(t, ch3_orig)
    features["delta_h"] = float(ch1_orig[-1] - ch1_orig[0])
    features["delta_v"] = float(ch3_orig[-1] - ch3_orig[0])
    
    # ---------------------------------------------------------
    # 2.3 サッカード活動度 (6次元)
    # ---------------------------------------------------------
    abs_mean_sac_h = float(np.mean(np.abs(ch2_orig)))
    abs_mean_sac_v = float(np.mean(np.abs(ch4_orig)))
    
    features["abs_mean_sac_h"] = abs_mean_sac_h
    features["abs_mean_sac_v"] = abs_mean_sac_v
    features["rms_sac_h"] = float(np.sqrt(np.mean(ch2_orig ** 2)))
    features["rms_sac_v"] = float(np.sqrt(np.mean(ch4_orig ** 2)))
    features["n_sac_h"] = float(_count_peaks(ch2_orig))
    features["n_sac_v"] = float(_count_peaks(ch4_orig))
    
    # ---------------------------------------------------------
    # 2.4 統合・関係特徴 (7次元)
    # ---------------------------------------------------------
    features["gaze_radius"] = float(np.sqrt(mean_h ** 2 + mean_v ** 2))
    features["gaze_angle"] = float(np.arctan2(mean_v, mean_h))
    
    sac_energy_h = abs_mean_sac_h
    sac_energy_v = abs_mean_sac_v
    sac_energy_total = sac_energy_h + sac_energy_v + EPSILON
    
    features["sac_energy_h"] = sac_energy_h
    features["sac_energy_v"] = sac_energy_v
    features["sac_energy_total"] = sac_energy_total
    features["sac_energy_ratio_h"] = sac_energy_h / sac_energy_total
    features["sac_energy_ratio_v"] = sac_energy_v / sac_energy_total
    
    # ---------------------------------------------------------
    # 2.5 サッカード時間局在 (4次元)
    # ---------------------------------------------------------
    features["energy_sac_h_first"] = float(np.mean(np.abs(ch2_orig[:half])))
    features["energy_sac_h_last"] = float(np.mean(np.abs(ch2_orig[half:])))
    features["energy_sac_v_first"] = float(np.mean(np.abs(ch4_orig[:half])))
    features["energy_sac_v_last"] = float(np.mean(np.abs(ch4_orig[half:])))
    
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
        特徴量辞書（29次元）
    
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
        特徴量DataFrame（29次元 + label列）
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
    リアルタイム特徴量抽出器（29次元）
    
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
    """特徴量カラム名のリストを取得（29次元）"""
    return FEATURE_COLUMNS.copy()
