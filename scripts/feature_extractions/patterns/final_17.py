"""
EOG 4ch 特徴量セット（26次元）

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
    5. 過去サッカード状態 (2次元): 直近サッカードの方向性（時間減衰付き）

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

過去サッカード状態の設計方針:
    last_sac_sign_h/v は直近サッカードの符号（+1/-1）に指数減衰を乗じた値。
    減衰率は100ウィンドウで概ねゼロ（0.01以下）になるよう設定。
    ウィンドウ内でサッカードが検出されない場合は状態を更新せず保持する。
    初回および未検出時は0。
    サッカード検出は差分系列の絶対値がウィンドウ内標準偏差の
    SACCADE_PEAK_THRESHOLD_STD 倍を超える局所最大値として定義する。
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
# 特徴量定義（26次元）
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
    # 2.5 過去サッカード状態 (2)
    # 直近サッカードの符号に指数減衰を乗じた値
    # 未検出・初回は 0
    "last_sac_sign_h",
    "last_sac_sign_v",
]


# ============================================================
# パラメータ
# ============================================================

# ゼロ除算防止用の小さな値
EPSILON = 1e-8

# サッカードピーク検出の閾値（差分系列の標準偏差の倍数）
SACCADE_PEAK_THRESHOLD_STD = 2.0

# 指数減衰率: 100ウィンドウで exp(-decay * 100) ≈ 0.01
# decay = -ln(0.01) / 100 ≈ 0.046
SAC_DECAY_RATE = 0.046


# ============================================================
# サッカード検出
# ============================================================

def _detect_last_sac_sign(diff_data: np.ndarray) -> float | None:
    """
    差分系列から最後のサッカードの符号を検出する

    閾値を超える局所最大値（絶対値）を検出し、
    その中で最後に発生したものの元の符号を返す。
    検出されない場合は None を返す。

    Args:
        diff_data: サッカード成分の一階差分系列

    Returns:
        最後のサッカードの符号（+1.0 / -1.0）または None
    """
    threshold = np.std(diff_data) * SACCADE_PEAK_THRESHOLD_STD
    abs_data = np.abs(diff_data)
    peaks, _ = signal.find_peaks(abs_data, height=threshold)

    if len(peaks) == 0:
        return None

    last_peak_idx = peaks[-1]
    return float(np.sign(diff_data[last_peak_idx]))


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(
    channels: dict[str, np.ndarray],
    prev_state: dict,
) -> tuple[dict[str, float], dict]:
    """
    チャンネルデータから26次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
        prev_state: 前ウィンドウからの状態辞書
            - "last_sac_sign_h": 水平方向直近サッカード符号（減衰前）
            - "last_sac_sign_v": 垂直方向直近サッカード符号（減衰前）
            - "windows_since_last_sac_h": 水平方向最後のサッカードからの経過ウィンドウ数
            - "windows_since_last_sac_v": 垂直方向最後のサッカードからの経過ウィンドウ数

    Returns:
        (特徴量辞書（26次元）, 更新後の状態辞書)
    """
    ch1 = channels["ch1"]  # 横方向位置
    ch2 = channels["ch2"]  # 横方向サッカード
    ch3 = channels["ch3"]  # 縦方向位置
    ch4 = channels["ch4"]  # 縦方向サッカード

    n_samples = len(ch1)
    t = np.arange(n_samples)
    half_pos = n_samples // 2

    # サッカード成分の一階差分（定数オフセット除去）
    diff_sac_h = np.diff(ch2)
    diff_sac_v = np.diff(ch4)
    n_diff = len(diff_sac_h)
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
    # ---------------------------------------------------------
    std_diff_sac_h = float(np.std(diff_sac_h))
    std_diff_sac_v = float(np.std(diff_sac_v))

    norm_h = rms_diff_sac_h / (std_diff_sac_h + EPSILON)
    norm_v = rms_diff_sac_v / (std_diff_sac_v + EPSILON)
    norm_total = norm_h + norm_v + EPSILON

    features["sac_relative_ratio_h"] = norm_h / norm_total
    features["sac_relative_ratio_v"] = norm_v / norm_total

    # ---------------------------------------------------------
    # 2.5 過去サッカード状態 (2次元)
    # ---------------------------------------------------------
    # 現在ウィンドウでのサッカード検出
    detected_sign_h = _detect_last_sac_sign(diff_sac_h)
    detected_sign_v = _detect_last_sac_sign(diff_sac_v)

    # 状態の更新
    new_state = dict(prev_state)

    if detected_sign_h is not None:
        new_state["last_sac_sign_h"] = detected_sign_h
        new_state["windows_since_last_sac_h"] = 0
    else:
        new_state["windows_since_last_sac_h"] = (
            prev_state["windows_since_last_sac_h"] + 1
        )

    if detected_sign_v is not None:
        new_state["last_sac_sign_v"] = detected_sign_v
        new_state["windows_since_last_sac_v"] = 0
    else:
        new_state["windows_since_last_sac_v"] = (
            prev_state["windows_since_last_sac_v"] + 1
        )

    # 減衰付き符号の出力
    features["last_sac_sign_h"] = new_state["last_sac_sign_h"] * float(
        np.exp(-SAC_DECAY_RATE * new_state["windows_since_last_sac_h"])
    )
    features["last_sac_sign_v"] = new_state["last_sac_sign_v"] * float(
        np.exp(-SAC_DECAY_RATE * new_state["windows_since_last_sac_v"])
    )

    return features, new_state


def _initial_state() -> dict:
    """過去サッカード状態の初期値（未検出・0埋め）"""
    return {
        "last_sac_sign_h": 0.0,
        "last_sac_sign_v": 0.0,
        "windows_since_last_sac_h": 0,
        "windows_since_last_sac_v": 0,
    }


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
        特徴量辞書（26次元、last_sac_sign_h/v は 0 固定）

    Raises:
        ValueError: サンプルが空の場合
    """
    if not samples:
        raise ValueError("samples must not be empty")

    channels = extract_channels(samples, channel_keys)
    features, _ = _compute_features(channels, _initial_state())
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

    Args:
        df: 入力DataFrame（各行が1サンプル）
        window_size: 窓サイズ
        stride: ストライド（Noneの場合はwindow_size）
        channel_columns: チャンネル列名
        label_column: ラベル列名（Noneならラベルなし）
        label_strategy: ラベル決定戦略

    Returns:
        特徴量DataFrame（26次元 + label列）
    """
    if stride is None:
        stride = window_size

    results = []
    state = _initial_state()

    for start, window_df in create_windows(df, window_size, stride):
        channels = {
            col: window_df[col].values.astype(np.float64)
            for col in channel_columns
        }

        features, state = _compute_features(channels, state)

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
    リアルタイム特徴量抽出器（26次元）

    過去サッカード状態をインスタンス変数として保持し、
    ウィンドウをまたいで直近サッカードの符号と経過ウィンドウ数を引き継ぐ。

    Example:
        >>> extractor = RealtimeExtractor(window_size=125, stride=62)
        >>> for sample in stream:
        ...     features = extractor.push(sample)
        ...     if features is not None:
        ...         prediction = model.predict([features])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sac_state: dict = _initial_state()

    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """特徴量計算（状態をインスタンス変数で管理）"""
        features, self._sac_state = _compute_features(channels, self._sac_state)
        return features


# ============================================================
# ユーティリティ
# ============================================================

def get_feature_columns() -> list[str]:
    """特徴量カラム名のリストを取得（26次元）"""
    return FEATURE_COLUMNS.copy()