"""
EOG 4ch 特徴量セット（69次元）

チャンネル構成:
    - ch1: 水平位置（水平EOG・低周波）
    - ch2: 水平サッカード成分
    - ch3: 垂直位置（垂直EOG・低周波）
    - ch4: 垂直サッカード成分

対象クラス: 「上」「下」「左」「右」「中央」「移動中」

特徴量カテゴリ（docs/context.md 準拠）:
    1. 位置チャンネル統計 (ch1, ch3): 16次元
    2. 位置トレンド・差分 (ch1, ch3): 10次元
    3. 2次元位置ベクトル特徴: 7次元
    4. サッカードチャンネル統計 (ch2, ch4): 10次元
    5. サッカードイベント特徴（ピークベース）: 8次元
    6. サッカード時間局在: 6次元
    7. チャンネル間関係・統合特徴: 4次元
    8. 時間・インデックス由来特徴: 4次元
    9. 周波数ドメイン特徴（簡易）: 4次元
    合計: 69次元
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
# 特徴量定義（69次元）
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # 1.1 水平位置 ch1 (8)
    "feat_mean_h",
    "feat_median_h",
    "feat_min_h",
    "feat_max_h",
    "feat_range_h",
    "feat_std_h",
    "feat_p25_h",
    "feat_p75_h",
    # 1.2 垂直位置 ch3 (8)
    "feat_mean_v",
    "feat_median_v",
    "feat_min_v",
    "feat_max_v",
    "feat_range_v",
    "feat_std_v",
    "feat_p25_v",
    "feat_p75_v",
    # 2.1 線形トレンド (6)
    "feat_slope_h",
    "feat_slope_v",
    "feat_r2_h",
    "feat_r2_v",
    "feat_delta_h",
    "feat_delta_v",
    # 2.2 差分統計 (4)
    "feat_mean_dh",
    "feat_std_dh",
    "feat_mean_dv",
    "feat_std_dv",
    # 2.3 ゼロ交差数 (2) ※差分のゼロ交差（トレンド変化回数）
    "feat_zc_dh",
    "feat_zc_dv",
    # 3. 2次元位置ベクトル特徴 (7)
    "feat_gaze_radius",
    "feat_gaze_angle",
    "feat_start_radius",
    "feat_end_radius",
    "feat_radius_change",
    "feat_start_angle",
    "feat_end_angle",
    # 4.1 水平サッカード ch2 (5)
    "feat_abs_mean_sac_h",
    "feat_rms_sac_h",
    "feat_std_sac_h",
    "feat_min_sac_h",
    "feat_max_sac_h",
    # 4.2 垂直サッカード ch4 (5)
    "feat_abs_mean_sac_v",
    "feat_rms_sac_v",
    "feat_std_sac_v",
    "feat_min_sac_v",
    "feat_max_sac_v",
    # 5.1 水平サッカードイベント (4)
    "feat_n_sac_h",
    "feat_max_amp_sac_h",
    "feat_mean_amp_sac_h",
    "feat_mean_isi_sac_h",
    # 5.2 垂直サッカードイベント (4)
    "feat_n_sac_v",
    "feat_max_amp_sac_v",
    "feat_mean_amp_sac_v",
    "feat_mean_isi_sac_v",
    # 6.1 水平サッカード時間局在 (3)
    "feat_energy_sac_h_first",
    "feat_energy_sac_h_last",
    "feat_diff_energy_sac_h",
    # 6.2 垂直サッカード時間局在 (3)
    "feat_energy_sac_v_first",
    "feat_energy_sac_v_last",
    "feat_diff_energy_sac_v",
    # 7. チャンネル間関係・統合特徴 (4)
    "feat_corr_pos_sac_h",
    "feat_corr_pos_sac_v",
    "feat_sac_energy_ratio_h",
    "feat_sac_energy_ratio_v",
    # 8. 時間・インデックス由来特徴 (4)
    "feat_t_step",
    "feat_t_norm",
    "feat_sin_2pi_t",
    "feat_cos_2pi_t",
    # 9. 周波数ドメイン特徴 (4)
    "feat_low_band_pow_h",
    "feat_low_band_pow_v",
    "feat_spec_centroid_h",
    "feat_spec_centroid_v",
]


# ============================================================
# パラメータ
# ============================================================

# サッカードピーク検出の閾値（標準偏差の倍数）
SACCADE_PEAK_THRESHOLD_STD = 2.0

# ゼロ除算防止用の小さな値
EPSILON = 1e-8

# 周波数ドメイン: 低周波帯域 (Hz)
LOW_BAND_HZ_MIN = 0.1
LOW_BAND_HZ_MAX = 5.0

# デフォルトサンプリングレート（Hz）
# バッチ処理時に外部から渡せない場合のフォールバック
DEFAULT_SAMPLING_RATE_HZ = 250


# ============================================================
# 内部ユーティリティ
# ============================================================

def _compute_slope_r2(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    単回帰の傾き (slope) と決定係数 (R²) を計算

    Returns:
        (slope, r2)
    """
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))

    ss_t = float(np.sum((t - t_mean) ** 2))
    ss_ty = float(np.sum((t - t_mean) * (y - y_mean)))
    ss_y = float(np.sum((y - y_mean) ** 2))

    if ss_t < EPSILON:
        return 0.0, 0.0

    slope = ss_ty / ss_t
    ss_res = float(np.sum((y - (slope * (t - t_mean) + y_mean)) ** 2))
    r2 = 1.0 - ss_res / (ss_y + EPSILON)

    return float(slope), float(r2)


def _count_zero_crossings(data: np.ndarray) -> int:
    """符号変化の数（ゼロ交差数）をカウント"""
    if len(data) < 2:
        return 0
    signs = np.sign(data)
    # ゼロは直前の符号を引き継ぐ（厳密なゼロ交差）
    signs[signs == 0] = 1
    return int(np.sum(signs[1:] != signs[:-1]))


def _detect_peaks(data: np.ndarray) -> np.ndarray:
    """
    絶対値に対して標準偏差ベースの閾値でピーク検出

    Returns:
        ピークインデックスの配列
    """
    abs_data = np.abs(data)
    threshold = np.std(abs_data) * SACCADE_PEAK_THRESHOLD_STD
    peaks, _ = signal.find_peaks(abs_data, height=threshold)
    return peaks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """標本相関係数（ゼロ除算ガード付き）"""
    if len(x) < 2:
        return 0.0
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x < EPSILON or std_y < EPSILON:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(
    channels: dict[str, np.ndarray],
    window_idx: int = 0,
    total_windows: int = 1,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> dict[str, float]:
    """
    チャンネルデータから69次元の特徴量を計算

    Args:
        channels: {"ch1": np.array([...]), "ch2": ..., "ch3": ..., "ch4": ...}
        window_idx: 現在のウィンドウインデックス（0始まり）
        total_windows: 全ウィンドウ数（時間正規化に使用）
        sampling_rate_hz: サンプリングレート (Hz)（周波数特徴に使用）

    Returns:
        特徴量辞書（69次元）
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
    # 1.1 水平位置統計 (8)
    # ----------------------------------------------------------
    features["feat_mean_h"]   = float(np.mean(ch1))
    features["feat_median_h"] = float(np.median(ch1))
    features["feat_min_h"]    = float(np.min(ch1))
    features["feat_max_h"]    = float(np.max(ch1))
    features["feat_range_h"]  = float(np.max(ch1) - np.min(ch1))
    features["feat_std_h"]    = float(np.std(ch1))
    features["feat_p25_h"]    = float(np.percentile(ch1, 25))
    features["feat_p75_h"]    = float(np.percentile(ch1, 75))

    # ----------------------------------------------------------
    # 1.2 垂直位置統計 (8)
    # ----------------------------------------------------------
    features["feat_mean_v"]   = float(np.mean(ch3))
    features["feat_median_v"] = float(np.median(ch3))
    features["feat_min_v"]    = float(np.min(ch3))
    features["feat_max_v"]    = float(np.max(ch3))
    features["feat_range_v"]  = float(np.max(ch3) - np.min(ch3))
    features["feat_std_v"]    = float(np.std(ch3))
    features["feat_p25_v"]    = float(np.percentile(ch3, 25))
    features["feat_p75_v"]    = float(np.percentile(ch3, 75))

    # ----------------------------------------------------------
    # 2.1 線形トレンド (6)
    # ----------------------------------------------------------
    slope_h, r2_h = _compute_slope_r2(t, ch1)
    slope_v, r2_v = _compute_slope_r2(t, ch3)

    features["feat_slope_h"] = slope_h
    features["feat_slope_v"] = slope_v
    features["feat_r2_h"]    = r2_h
    features["feat_r2_v"]    = r2_v
    features["feat_delta_h"] = float(ch1[-1] - ch1[0])
    features["feat_delta_v"] = float(ch3[-1] - ch3[0])

    # ----------------------------------------------------------
    # 2.2 差分統計 (4)
    # ----------------------------------------------------------
    dch1 = np.diff(ch1)
    dch3 = np.diff(ch3)

    features["feat_mean_dh"] = float(np.mean(dch1))
    features["feat_std_dh"]  = float(np.std(dch1))
    features["feat_mean_dv"] = float(np.mean(dch3))
    features["feat_std_dv"]  = float(np.std(dch3))

    # 2.3 ゼロ交差数 (2)
    features["feat_zc_dh"] = float(_count_zero_crossings(dch1))
    features["feat_zc_dv"] = float(_count_zero_crossings(dch3))

    # ----------------------------------------------------------
    # 3. 2次元位置ベクトル特徴 (7)
    # ----------------------------------------------------------
    mu_h = features["feat_mean_h"]
    mu_v = features["feat_mean_v"]
    h_start, v_start = float(ch1[0]),  float(ch3[0])
    h_end,   v_end   = float(ch1[-1]), float(ch3[-1])

    start_radius = float(np.sqrt(h_start ** 2 + v_start ** 2))
    end_radius   = float(np.sqrt(h_end   ** 2 + v_end   ** 2))

    features["feat_gaze_radius"]   = float(np.sqrt(mu_h ** 2 + mu_v ** 2))
    features["feat_gaze_angle"]    = float(np.arctan2(mu_v, mu_h))
    features["feat_start_radius"]  = start_radius
    features["feat_end_radius"]    = end_radius
    features["feat_radius_change"] = end_radius - start_radius
    features["feat_start_angle"]   = float(np.arctan2(v_start, h_start))
    features["feat_end_angle"]     = float(np.arctan2(v_end,   h_end))

    # ----------------------------------------------------------
    # 4.1 水平サッカード統計 (5)
    # ----------------------------------------------------------
    features["feat_abs_mean_sac_h"] = float(np.mean(np.abs(ch2)))
    features["feat_rms_sac_h"]      = float(np.sqrt(np.mean(ch2 ** 2)))
    features["feat_std_sac_h"]      = float(np.std(ch2))
    features["feat_min_sac_h"]      = float(np.min(ch2))
    features["feat_max_sac_h"]      = float(np.max(ch2))

    # ----------------------------------------------------------
    # 4.2 垂直サッカード統計 (5)
    # ----------------------------------------------------------
    features["feat_abs_mean_sac_v"] = float(np.mean(np.abs(ch4)))
    features["feat_rms_sac_v"]      = float(np.sqrt(np.mean(ch4 ** 2)))
    features["feat_std_sac_v"]      = float(np.std(ch4))
    features["feat_min_sac_v"]      = float(np.min(ch4))
    features["feat_max_sac_v"]      = float(np.max(ch4))

    # ----------------------------------------------------------
    # 5. サッカードイベント特徴（ピークベース）(8)
    # ----------------------------------------------------------
    peaks_h = _detect_peaks(ch2)
    peaks_v = _detect_peaks(ch4)

    # 水平 (4)
    features["feat_n_sac_h"] = float(len(peaks_h))
    if len(peaks_h) > 0:
        peak_amps_h = ch2[peaks_h]
        features["feat_max_amp_sac_h"]  = float(np.max(peak_amps_h))
        features["feat_mean_amp_sac_h"] = float(np.mean(peak_amps_h))
        isi_h = np.diff(peaks_h)
        features["feat_mean_isi_sac_h"] = float(np.mean(isi_h)) if len(isi_h) > 0 else 0.0
    else:
        features["feat_max_amp_sac_h"]  = 0.0
        features["feat_mean_amp_sac_h"] = 0.0
        features["feat_mean_isi_sac_h"] = 0.0

    # 垂直 (4)
    features["feat_n_sac_v"] = float(len(peaks_v))
    if len(peaks_v) > 0:
        peak_amps_v = ch4[peaks_v]
        features["feat_max_amp_sac_v"]  = float(np.max(peak_amps_v))
        features["feat_mean_amp_sac_v"] = float(np.mean(peak_amps_v))
        isi_v = np.diff(peaks_v)
        features["feat_mean_isi_sac_v"] = float(np.mean(isi_v)) if len(isi_v) > 0 else 0.0
    else:
        features["feat_max_amp_sac_v"]  = 0.0
        features["feat_mean_amp_sac_v"] = 0.0
        features["feat_mean_isi_sac_v"] = 0.0

    # ----------------------------------------------------------
    # 6. サッカード時間局在 (6)
    # ----------------------------------------------------------
    e_h_first = float(np.mean(np.abs(ch2[:half])))
    e_h_last  = float(np.mean(np.abs(ch2[half:])))
    e_v_first = float(np.mean(np.abs(ch4[:half])))
    e_v_last  = float(np.mean(np.abs(ch4[half:])))

    features["feat_energy_sac_h_first"] = e_h_first
    features["feat_energy_sac_h_last"]  = e_h_last
    features["feat_diff_energy_sac_h"]  = e_h_last - e_h_first
    features["feat_energy_sac_v_first"] = e_v_first
    features["feat_energy_sac_v_last"]  = e_v_last
    features["feat_diff_energy_sac_v"]  = e_v_last - e_v_first

    # ----------------------------------------------------------
    # 7. チャンネル間関係・統合特徴 (4)
    # ----------------------------------------------------------
    features["feat_corr_pos_sac_h"] = _corr(ch1, ch2)
    features["feat_corr_pos_sac_v"] = _corr(ch3, ch4)

    sac_e_h = features["feat_abs_mean_sac_h"]
    sac_e_v = features["feat_abs_mean_sac_v"]
    sac_e_total = sac_e_h + sac_e_v + EPSILON
    features["feat_sac_energy_ratio_h"] = sac_e_h / sac_e_total
    features["feat_sac_energy_ratio_v"] = sac_e_v / sac_e_total

    # ----------------------------------------------------------
    # 8. 時間・インデックス由来特徴 (4)
    # ----------------------------------------------------------
    w = float(window_idx)
    w_max = float(max(total_windows - 1, 1))
    t_norm = w / w_max

    features["feat_t_step"]      = w
    features["feat_t_norm"]      = t_norm
    features["feat_sin_2pi_t"]   = float(np.sin(2.0 * np.pi * t_norm))
    features["feat_cos_2pi_t"]   = float(np.cos(2.0 * np.pi * t_norm))

    # ----------------------------------------------------------
    # 9. 周波数ドメイン特徴 (4)
    # ----------------------------------------------------------
    n_fft = int(2 ** np.ceil(np.log2(n)))  # n の次の 2 の冪

    f1 = np.fft.rfft(ch1, n=n_fft)
    f3 = np.fft.rfft(ch3, n=n_fft)
    p1 = np.abs(f1) ** 2
    p3 = np.abs(f3) ** 2

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz)
    low_mask = (freqs >= LOW_BAND_HZ_MIN) & (freqs <= LOW_BAND_HZ_MAX)

    features["feat_low_band_pow_h"]   = float(np.sum(p1[low_mask]))
    features["feat_low_band_pow_v"]   = float(np.sum(p3[low_mask]))
    features["feat_spec_centroid_h"]  = float(
        np.sum(freqs * p1) / (np.sum(p1) + EPSILON)
    )
    features["feat_spec_centroid_v"]  = float(
        np.sum(freqs * p3) / (np.sum(p3) + EPSILON)
    )

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
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> dict[str, float]:
    """
    窓内のサンプルから特徴量を抽出

    Args:
        samples: 窓サイズ分のサンプルリスト
        channel_keys: チャンネルのキー名
        window_idx: ウィンドウインデックス（時間特徴に使用）
        total_windows: 全ウィンドウ数（時間特徴に使用）
        sampling_rate_hz: サンプリングレート（周波数特徴に使用）

    Returns:
        特徴量辞書（69次元）
    """
    if not samples:
        raise ValueError("samples must not be empty")

    channels = extract_channels(samples, channel_keys)
    return _compute_features(
        channels,
        window_idx=window_idx,
        total_windows=total_windows,
        sampling_rate_hz=sampling_rate_hz,
    )


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
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
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
        sampling_rate_hz: サンプリングレート（周波数特徴に使用）

    Returns:
        特徴量DataFrame（69次元 [+ label列]）
    """
    if stride is None:
        stride = window_size

    # 総ウィンドウ数を事前計算
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
            sampling_rate_hz=sampling_rate_hz,
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
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
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
        sampling_rate_hz: サンプリングレート（周波数特徴に使用）

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
        sampling_rate_hz=sampling_rate_hz,
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
    リアルタイム特徴量抽出器（69次元）

    時間特徴（feat_t_step 等）はウィンドウが抽出されるたびに
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
        sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    ):
        super().__init__(window_size, stride, channel_keys=channel_keys)
        self._sampling_rate_hz = sampling_rate_hz
        self._window_count = 0

    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """特徴量計算（ウィンドウカウンタを使用）"""
        features = _compute_features(
            channels,
            window_idx=self._window_count,
            total_windows=1,  # リアルタイムでは全数不明: t_norm=0固定
            sampling_rate_hz=self._sampling_rate_hz,
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
    """特徴量カラム名のリストを取得（69次元）"""
    return FEATURE_COLUMNS.copy()
