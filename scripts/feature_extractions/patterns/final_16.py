"""
EOG 4ch 特徴量セット（25次元）

チャンネル構成:
    - ch1: 横方向位置成分（水平 EOG・低周波）
    - ch2: 横方向サッカード成分（水平サッカード時スパイク）
    - ch3: 縦方向位置成分（垂直 EOG・低周波）
    - ch4: 縦方向サッカード成分（垂直サッカード時スパイク）

対象クラス: 「上」「下」「左」「右」「中央」「移動中」

特徴量カテゴリ:
    1. 位置レベル・分布 (10次元): 保持位置判別の主軸
    2. 位置トレンド・変化量 (6次元): 移動中の検出・タイミング
    3. サッカード活動度・方向・局在 (6次元): 移動中の識別・方向性・タイミング
    4. 統合・関係特徴 (2次元): サッカードの軸方向の相対的優勢
    5. 状態記憶特徴 (1次元): 過去のサッカード履歴

オフセット設計方針:
    位置成分（ch1/ch3）は教師・タスク間でベースラインが一致しているため、
    定数オフセットを含んだまま統計量を計算してよい。
    range, std, slope, skew はいずれも定数オフセット非依存である。

    サッカード成分（ch2/ch4）はオフセットが不明であるため、
    全ての特徴量を一階差分系列（np.diff）から計算する。
    差分系列は定数オフセットを完全に除去する。
    差分系列のサンプル数は元信号より1少ない（n-1）点に注意。

縦横スケール設計方針:
    縦横チャンネル間でオフセットおよび振幅スケールが一致しない。
    sac_relative_ratio_h/v は各チャンネルの差分系列内での相対的な傾斜
    （rms_diff / std_diff）を基底として比率を計算し、縦横スケール差を吸収する。

状態記憶設計方針:
    last_sac_axis は過去ウィンドウを含む直近のサッカード検出結果を保持する。
    サッカード検出は差分系列の絶対値がウィンドウ内標準偏差の
    SACCADE_PEAK_THRESHOLD_STD 倍を超える点の存在で判定する。
    現在ウィンドウでサッカードが検出されない場合は前回値を維持する。
    初回および一度もサッカードが検出されていない場合は 0。

    バッチ処理では batch_extract_from_dataframe 内でウィンドウ順に
    状態を引き継ぐ。リアルタイム処理では RealtimeExtractor のインスタンス
    変数として状態を保持する。
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal, stats
import pandas as pd

from .. import (
    BaseRealtimeExtractor,
    create_windows,
    determine_label,
    extract_channels,
)


# ============================================================
# 特徴量定義（25次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 2.1 位置レベル・分布 (10)
    "mean_h",
    "mean_v",
    "median_h",
    "median_v",
    "range_h",
    "range_v",
    "std_h",
    "std_v",
    "skew_h",
    "skew_v",
    # 2.2 位置トレンド・変化量 (6)
    "slope_h",
    "slope_v",
    "slope_h_first",
    "slope_h_last",
    "slope_v_first",
    "slope_v_last",
    # 2.3 サッカード活動度・方向・局在 (6)
    "rms_diff_sac_h",
    "rms_diff_sac_v",
    "mean_diff_sac_h_first",
    "mean_diff_sac_h_last",
    "mean_diff_sac_v_first",
    "mean_diff_sac_v_last",
    # 2.4 統合・関係特徴 (2)
    "sac_relative_ratio_h",
    "sac_relative_ratio_v",
    # 2.5 状態記憶特徴 (1)
    # +1 = 水平が最後 / -1 = 垂直が最後 / 0 = 未検出
    "last_sac_axis",
]


# ============================================================
# パラメータ
# ============================================================

# サッカード検出閾値（差分系列の標準偏差の倍数）
SACCADE_PEAK_THRESHOLD_STD = 2.0

# ゼロ除算防止用の小さな値
EPSILON = 1e-8


# ============================================================
# サッカード検出
# ============================================================

def _detect_last_sac_axis(
    diff_sac_h: np.ndarray,
    diff_sac_v: np.ndarray,
) -> float | None:
    """
    差分系列から最後に発生したサッカードの軸を検出する

    検出基準: 差分系列の絶対値がウィンドウ内標準偏差の
              SACCADE_PEAK_THRESHOLD_STD 倍を超える局所最大値

    Args:
        diff_sac_h: 水平サッカード成分の差分系列
        diff_sac_v: 垂直サッカード成分の差分系列

    Returns:
        +1.0 = 水平が最後 / -1.0 = 垂直が最後 / None = 両方未検出
    """
    last_h = _last_peak_index(diff_sac_h)
    last_v = _last_peak_index(diff_sac_v)

    if last_h is None and last_v is None:
        return None
    if last_h is None:
        return -1.0
    if last_v is None:
        return 1.0
    return 1.0 if last_h >= last_v else -1.0


def _last_peak_index(diff: np.ndarray) -> int | None:
    """
    差分系列内で閾値を超える最後の局所最大値のインデックスを返す

    閾値: 差分系列の標準偏差 × SACCADE_PEAK_THRESHOLD_STD
    検出対象: 正負両方のスパイク（絶対値系列に対してピーク検出）
    """
    threshold = np.std(diff) * SACCADE_PEAK_THRESHOLD_STD
    abs_diff = np.abs(diff)
    peaks, _ = signal.find_peaks(abs_diff, height=threshold)

    if len(peaks) == 0:
        return None
    return int(peaks[-1])


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(
    channels: dict[str, np.ndarray],
    last_sac_axis: float = 0.0,
) -> tuple[dict[str, float], float]:
    """
    チャンネルデータから25次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
        last_sac_axis: 直前までの最後のサッカード軸の状態
                       (+1 = 水平 / -1 = 垂直 / 0 = 未検出)

    Returns:
        (特徴量辞書, 更新後の last_sac_axis)
    """
    ch1 = channels["ch1"]  # 横方向位置
    ch2 = channels["ch2"]  # 横方向サッカード
    ch3 = channels["ch3"]  # 縦方向位置
    ch4 = channels["ch4"]  # 縦方向サッカード

    n_samples = len(ch1)
    t = np.arange(n_samples)
    half_pos = n_samples // 2

    # サッカード成分の一階差分（定数オフセット除去）
    # サンプル数は n_samples - 1 になる
    diff_sac_h = np.diff(ch2)
    diff_sac_v = np.diff(ch4)
    n_diff = len(diff_sac_h)  # n_samples - 1
    half_diff = n_diff // 2

    features = {}

    # ---------------------------------------------------------
    # 2.1 位置レベル・分布 (10次元)
    # ---------------------------------------------------------
    features["mean_h"] = float(np.mean(ch1))
    features["mean_v"] = float(np.mean(ch3))
    features["median_h"] = float(np.median(ch1))
    features["median_v"] = float(np.median(ch3))
    features["range_h"] = float(np.max(ch1) - np.min(ch1))
    features["range_v"] = float(np.max(ch3) - np.min(ch3))
    features["std_h"] = float(np.std(ch1))
    features["std_v"] = float(np.std(ch3))
    features["skew_h"] = float(stats.skew(ch1))
    features["skew_v"] = float(stats.skew(ch3))

    # ---------------------------------------------------------
    # 2.2 位置トレンド・変化量 (6次元)
    # ---------------------------------------------------------
    features["slope_h"] = _compute_slope(t, ch1)
    features["slope_v"] = _compute_slope(t, ch3)

    t_first = t[:half_pos]
    t_last = t[half_pos:]

    features["slope_h_first"] = _compute_slope(t_first, ch1[:half_pos])
    features["slope_h_last"] = _compute_slope(t_last, ch1[half_pos:])
    features["slope_v_first"] = _compute_slope(t_first, ch3[:half_pos])
    features["slope_v_last"] = _compute_slope(t_last, ch3[half_pos:])

    # ---------------------------------------------------------
    # 2.3 サッカード活動度・方向・局在 (6次元)
    # 全て差分系列から計算（定数オフセット非依存）
    # ---------------------------------------------------------
    rms_diff_sac_h = float(np.sqrt(np.mean(diff_sac_h ** 2)))
    rms_diff_sac_v = float(np.sqrt(np.mean(diff_sac_v ** 2)))

    features["rms_diff_sac_h"] = rms_diff_sac_h
    features["rms_diff_sac_v"] = rms_diff_sac_v
    features["mean_diff_sac_h_first"] = float(np.mean(diff_sac_h[:half_diff]))
    features["mean_diff_sac_h_last"] = float(np.mean(diff_sac_h[half_diff:]))
    features["mean_diff_sac_v_first"] = float(np.mean(diff_sac_v[:half_diff]))
    features["mean_diff_sac_v_last"] = float(np.mean(diff_sac_v[half_diff:]))

    # ---------------------------------------------------------
    # 2.4 統合・関係特徴 (2次元)
    # チャンネル内正規化: norm = rms_diff / (std_diff + EPSILON)
    # ---------------------------------------------------------
    std_diff_sac_h = float(np.std(diff_sac_h))
    std_diff_sac_v = float(np.std(diff_sac_v))

    norm_h = rms_diff_sac_h / (std_diff_sac_h + EPSILON)
    norm_v = rms_diff_sac_v / (std_diff_sac_v + EPSILON)
    norm_total = norm_h + norm_v + EPSILON

    features["sac_relative_ratio_h"] = norm_h / norm_total
    features["sac_relative_ratio_v"] = norm_v / norm_total

    # ---------------------------------------------------------
    # 2.5 状態記憶特徴 (1次元)
    # 現在ウィンドウでサッカードが検出された場合のみ状態を更新
    # ---------------------------------------------------------
    detected = _detect_last_sac_axis(diff_sac_h, diff_sac_v)
    if detected is not None:
        last_sac_axis = detected

    features["last_sac_axis"] = last_sac_axis

    return features, last_sac_axis


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


# ============================================================
# 単一窓の特徴量抽出
# ============================================================

def extract_features(
    samples: list[Any],
    *,
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
) -> dict[str, float]:
    """
    窓内のサンプルから特徴量を抽出（状態管理なし・単発呼び出し用）

    Args:
        samples: 窓サイズ分のサンプルリスト
        channel_keys: チャンネルのキー名

    Returns:
        特徴量辞書（25次元、last_sac_axis は 0 固定）

    Raises:
        ValueError: サンプルが空の場合
    """
    if not samples:
        raise ValueError("samples must not be empty")

    channels = extract_channels(samples, channel_keys)
    features, _ = _compute_features(channels)
    return features


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

    ウィンドウを順次処理し last_sac_axis の状態を引き継ぐ。
    初回ウィンドウは last_sac_axis = 0 で開始する。

    Args:
        df: 入力DataFrame（各行が1サンプル）
        window_size: 窓サイズ
        stride: ストライド（Noneの場合はwindow_size）
        channel_columns: チャンネル列名
        label_column: ラベル列名（Noneならラベルなし）
        label_strategy: ラベル決定戦略

    Returns:
        特徴量DataFrame（25次元 + label列）
    """
    if stride is None:
        stride = window_size

    results = []
    last_sac_axis: float = 0.0

    for start, window_df in create_windows(df, window_size, stride):
        channels = {
            col: window_df[col].values.astype(np.float64)
            for col in channel_columns
        }

        features, last_sac_axis = _compute_features(channels, last_sac_axis)

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
    リアルタイム特徴量抽出器（25次元）

    last_sac_axis をインスタンス変数として保持し、
    ウィンドウをまたいで状態を引き継ぐ。

    Example:
        >>> extractor = RealtimeExtractor(window_size=125, stride=62)
        >>> for sample in stream:
        ...     features = extractor.push(sample)
        ...     if features is not None:
        ...         prediction = model.predict([features])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sac_axis: float = 0.0

    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """特徴量計算（状態を引き継ぎながら更新）"""
        features, self._last_sac_axis = _compute_features(
            channels, self._last_sac_axis
        )
        return features


# ============================================================
# ユーティリティ
# ============================================================

def get_feature_columns() -> list[str]:
    """特徴量カラム名のリストを取得（25次元）"""
    return FEATURE_COLUMNS.copy()