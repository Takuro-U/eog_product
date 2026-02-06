"""
特徴量抽出テンプレート

新しい特徴量抽出パターンを作成する際は、このファイルをコピーして
以下を実装する:
    1. FEATURE_COLUMNS: 出力する特徴量カラム名のリスト
    2. _compute_features(): 特徴量計算ロジック

使用例:
    # バッチ処理（CSV → 特徴量DataFrame）
    from scripts.feature_extractions import your_extractor
    features_df = your_extractor.batch_extract_from_csv(
        "data/labeled/labeled_xxx.csv",
        window_size=125
    )
    
    # リアルタイム処理
    extractor = your_extractor.RealtimeExtractor(window_size=125)
    for sample in stream:
        features = extractor.push(sample)
        if features is not None:
            prediction = model.predict([features])
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import (
    BaseRealtimeExtractor,
    create_windows,
    determine_label,
    extract_channels,
)


# ============================================================
# 特徴量定義
# ============================================================

# TODO: 出力する特徴量カラム名を定義
FEATURE_COLUMNS: list[str] = [
    "placeholder",  # 削除して実際の特徴量名を記述
]


# ============================================================
# コア: 特徴量計算
# ============================================================

def _compute_features(channels: dict[str, np.ndarray]) -> dict[str, float]:
    """
    チャンネルデータから特徴量を計算
    
    Args:
        channels: {"ch1": np.array([...]), "ch2": np.array([...]), ...}
                  各配列は窓サイズ分のサンプル
    
    Returns:
        特徴量辞書 {"feature_name": value, ...}
    
    Note:
        FEATURE_COLUMNS と一致するキーを持つ辞書を返すこと
    """
    features = {}
    
    # ==========================================================
    # TODO: 特徴量計算を実装
    # 
    # 例:
    # for ch_name, ch_data in channels.items():
    #     features[f"mean_{ch_name}"] = float(np.mean(ch_data))
    #     features[f"std_{ch_name}"] = float(np.std(ch_data))
    # ==========================================================
    
    # プレースホルダー（削除予定）
    features["placeholder"] = 0.0
    
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
        特徴量辞書
    
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
        特徴量DataFrame
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
    リアルタイム特徴量抽出器
    
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
    """特徴量カラム名のリストを取得"""
    return FEATURE_COLUMNS.copy()
