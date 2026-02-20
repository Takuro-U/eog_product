"""
洗練版 EOG 4ch 特徴量セット（docs/context.md 準拠）

チャンネル構成:
    - ch1: 水平位置（水平EOG・低周波）
    - ch2: 水平サッカード成分
    - ch3: 垂直位置（垂直EOG・低周波）
    - ch4: 垂直サッカード成分

特徴量カテゴリ:
    1. 位置（方向・中央からの距離・揺らぎ）: 8次元
    2. 位置トレンド・変化: 4次元
    3. サッカード総量・方向性: 7次元
    4. サッカード時間局在（前半/後半）: 4次元
    5. 「移動中」スコア: 3次元
    6. 時間インデックス（任意）: 2次元
    合計: 28次元
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
# 特徴量定義（28次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 1. 位置（8）
    "feat_mean_h",
    "feat_mean_v",
    "feat_median_h",
    "feat_median_v",
    "feat_std_h",
    "feat_std_v",
    "feat_gaze_radius",
    "feat_gaze_angle",
    # 2. 位置トレンド・変化（4）
    "feat_slope_h",
    "feat_slope_v",
    "feat_delta_h",
    "feat_delta_v",
    # 3. サッカード総量・方向性（7）
    "feat_sac_energy_h",
    "feat_sac_energy_v",
    "feat_sac_energy_total",
    "feat_sac_energy_ratio_h",
    "feat_sac_energy_ratio_v",
    "feat_n_sac_h",
    "feat_n_sac_v",
    # 4. サッカード時間局在（4）
    "feat_energy_sac_h_first",
    "feat_energy_sac_h_last",
    "feat_energy_sac_v_first",
    "feat_energy_sac_v_last",
    # 5. 「移動中」スコア（3）
    "feat_motion_score_h",
    "feat_motion_score_v",
    "feat_motion_score",
    # 6. 時間インデックス（2）
    "feat_t_step",
    "feat_t_norm",
]


# ============================================================
# パラメータ
# ============================================================

# サッカードピーク検出の閾値（標準偏差の倍数）
SACCADE_PEAK_THRESHOLD_STD = 2.0

# ゼロ除算防止用の小さな値
EPSILON = 1e-8

# デフォルトサンプリングレート（Hz）
DEFAULT_SAMPLING_RATE_HZ = 250


# ============================================================
# 内部ユーティリティ
# ============================================================

def _compute_slope(t: np.ndarray, y: np.ndarray) -> float:
    """単回帰の傾きを計算（最小二乗法）"""
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    denom = float(np.sum((t - t_mean) ** 2))
    if denom < EPSILON:
        return 0.0
    return float(np.sum((t - t_mean) * (y - y_mean)) / denom)


def _detect_peaks(data: np.ndarray) -> np.ndarray:
    """絶対値に対して標準偏差ベースの閾値でピーク検出"""
    abs_data = np.abs(data)
    threshold = float(np.std(abs_data)) * SACCADE_PEAK_THRESHOLD_STD
    peaks, _ = signal.find_peaks(abs_data, height=threshold)
    return peaks


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(
    channels: dict[str, np.ndarray],
    window_idx: int = 0,
    total_windows: int = 1,
) -> dict[str, float]:
    """
    チャンネルデータから28次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
        window_idx: 現在のウィンドウインデックス（0始まり、時間特徴に使用）
        total_windows: 全ウィンドウ数（時間正規化に使用）

    Returns:
        特徴量辞書（28次元）
    """
    ch1 = channels["ch1"]  # 水平位置
    ch2 = channels["ch2"]  # 水平サッカード
    ch3 = channels["ch3"]  # 垂直位置
    ch4 = channels["ch4"]  # 垂直サッカード

    n = len(ch1)
    t = np.arange(n, dtype=np.float64)
    half = n // 2

    features: dict[str, float] = {}

    # ----------------------------------------------------------
    # 1. 位置（方向・中央からの距離・揺らぎ）(8)
    # ----------------------------------------------------------
    mean_h = float(np.mean(ch1))
    mean_v = float(np.mean(ch3))

    features["feat_mean_h"]     = mean_h
    features["feat_mean_v"]     = mean_v
    features["feat_median_h"]   = float(np.median(ch1))
    features["feat_median_v"]   = float(np.median(ch3))
    features["feat_std_h"]      = float(np.std(ch1))
    features["feat_std_v"]      = float(np.std(ch3))
    features["feat_gaze_radius"] = float(np.sqrt(mean_h ** 2 + mean_v ** 2))
    features["feat_gaze_angle"]  = float(np.arctan2(mean_v, mean_h))

    # ----------------------------------------------------------
    # 2. 位置トレンド・変化（4）
    # ----------------------------------------------------------
    slope_h = _compute_slope(t, ch1)
    slope_v = _compute_slope(t, ch3)

    features["feat_slope_h"] = slope_h
    features["feat_slope_v"] = slope_v
    features["feat_delta_h"] = float(ch1[-1] - ch1[0])
    features["feat_delta_v"] = float(ch3[-1] - ch3[0])

    # ----------------------------------------------------------
    # 3. サッカード総量・方向性（7）
    # ----------------------------------------------------------
    sac_energy_h = float(np.mean(np.abs(ch2)))
    sac_energy_v = float(np.mean(np.abs(ch4)))
    sac_energy_total = sac_energy_h + sac_energy_v + EPSILON

    features["feat_sac_energy_h"]       = sac_energy_h
    features["feat_sac_energy_v"]       = sac_energy_v
    features["feat_sac_energy_total"]   = sac_energy_total
    features["feat_sac_energy_ratio_h"] = sac_energy_h / sac_energy_total
    features["feat_sac_energy_ratio_v"] = sac_energy_v / sac_energy_total
    features["feat_n_sac_h"]            = float(len(_detect_peaks(ch2)))
    features["feat_n_sac_v"]            = float(len(_detect_peaks(ch4)))

    # ----------------------------------------------------------
    # 4. サッカード時間局在（前半/後半）(4)
    # ----------------------------------------------------------
    features["feat_energy_sac_h_first"] = float(np.mean(np.abs(ch2[:half])))
    features["feat_energy_sac_h_last"]  = float(np.mean(np.abs(ch2[half:])))
    features["feat_energy_sac_v_first"] = float(np.mean(np.abs(ch4[:half])))
    features["feat_energy_sac_v_last"]  = float(np.mean(np.abs(ch4[half:])))

    # ----------------------------------------------------------
    # 5. 「移動中」スコア（位置変化 × サッカード）(3)
    # ----------------------------------------------------------
    features["feat_motion_score_h"] = abs(slope_h) * sac_energy_h
    features["feat_motion_score_v"] = abs(slope_v) * sac_energy_v
    features["feat_motion_score"]   = (
        float(np.sqrt(slope_h ** 2 + slope_v ** 2)) * sac_energy_total
    )

    # ----------------------------------------------------------
    # 6. 時間インデックス（2）
    # ----------------------------------------------------------
    w_max = float(max(total_windows - 1, 1))
    t_norm = float(window_idx) / w_max

    features["feat_t_step"] = float(window_idx)
    features["feat_t_norm"] = t_norm

    return features


# ============================================================
# 単一窓の特徴量抽出
# ============================================================

def extract_features(
    samples: list[Any],
    *,
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    window_idx: int = 0,
    total_windows: int = 1,
) -> dict[str, float]:
    """
    窓内のサンプルから特徴量を抽出

    Args:
        samples: 窓サイズ分のサンプルリスト
        channel_keys: チャンネルのキー名
        window_idx: ウィンドウインデックス（時間特徴に使用）
        total_windows: 全ウィンドウ数（時間特徴に使用）

    Returns:
        特徴量辞書（28次元）
    """
    if not samples:
        raise ValueError("samples must not be empty")

    channels = extract_channels(samples, channel_keys)
    return _compute_features(channels, window_idx=window_idx, total_windows=total_windows)


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
        特徴量DataFrame（28次元 [+ label列]）
    """
    if stride is None:
        stride = window_size

    n = len(df)
    total_windows = max(1, (n - window_size) // stride + 1)

    results = []

    for window_idx, (_, window_df) in enumerate(
        create_windows(df, window_size, stride)
    ):
        channels = {
            col: window_df[col].values.astype(np.float64)
            for col in channel_columns
        }

        features = _compute_features(
            channels,
            window_idx=window_idx,
            total_windows=total_windows,
        )

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
    リアルタイム特徴量抽出器（28次元）

    時間特徴（feat_t_step/feat_t_norm）はウィンドウが抽出されるたびに
    内部カウンタで自動インクリメントされる。
    total_windows は未知なので feat_t_norm は 0 固定となる。

    Example:
        >>> extractor = RealtimeExtractor(window_size=25, stride=25)
        >>> for sample in stream:
        ...     features = extractor.push(sample)
        ...     if features is not None:
        ...         prediction = model.predict([list(features.values())])
    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        *,
        channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    ):
        super().__init__(window_size, stride, channel_keys=channel_keys)
        self._window_count = 0

    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """特徴量計算（ウィンドウカウンタを使用）"""
        features = _compute_features(
            channels,
            window_idx=self._window_count,
            total_windows=1,  # リアルタイムでは全数不明: t_norm=0固定
        )
        self._window_count += 1
        return features

    def reset(self) -> None:
        """バッファとカウンタをクリア"""
        super().reset()
        self._window_count = 0


# ============================================================
# ユーティリティ
# ============================================================

def get_feature_columns() -> list[str]:
    """特徴量カラム名のリストを取得（28次元）"""
    return FEATURE_COLUMNS.copy()
