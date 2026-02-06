"""
特徴量抽出モジュール

複数の特徴量抽出ロジックを格納するディレクトリ。
各特徴量抽出処理は 1ファイル = 1パターン で実装する。

構造:
    scripts/feature_extractions/
    ├── __init__.py          # 共通ユーティリティ + ディスパッチャー
    ├── _template.py         # 新規作成用テンプレート
    └── patterns/            # 個別パターン格納ディレクトリ
        ├── __init__.py
        └── eog_29dim.py     # パターン例: 29次元特徴量

使用例（パターン名を引数で指定）:
    from scripts import feature_extractions as fe
    
    # バッチ処理（パターン名を引数で指定）
    features_df = fe.batch_extract_from_csv(
        "data/labeled/labeled_xxx.csv",
        pattern="eog_29dim",
        window_size=125
    )
    
    # リアルタイム処理（パターン名を引数で指定）
    extractor = fe.create_realtime_extractor(
        pattern="eog_29dim",
        window_size=125
    )
    for sample in stream:
        features = extractor.push(sample)
        if features is not None:
            prediction = model.predict([features])
    
    # 利用可能なパターン一覧
    patterns = fe.list_patterns()
    
    # 特徴量カラム名取得
    columns = fe.get_feature_columns("eog_29dim")
"""

from collections import deque
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType


# ============================================================
# 型定義
# ============================================================

@runtime_checkable
class SampleLike(Protocol):
    """サンプルデータのプロトコル（duck typing用）"""
    ch1: int | float
    ch2: int | float
    ch3: int | float
    ch4: int | float


# ============================================================
# 共通ユーティリティ: チャンネルデータ抽出
# ============================================================

def extract_channels(
    samples: list[Any],
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
) -> dict[str, np.ndarray]:
    """
    サンプルリストから各チャンネルのデータを抽出
    
    様々な入力形式に対応:
    - NamedTuple (AdcSample)
    - dict
    - 属性アクセス可能なオブジェクト
    
    Args:
        samples: サンプルのリスト
        channel_keys: チャンネルのキー名
    
    Returns:
        {"ch1": np.array([...]), "ch2": np.array([...]), ...}
    """
    channels = {key: [] for key in channel_keys}
    
    for sample in samples:
        for key in channel_keys:
            if isinstance(sample, dict):
                value = sample[key]
            elif hasattr(sample, key):
                value = getattr(sample, key)
            else:
                raise TypeError(
                    f"Sample must be dict or have attribute '{key}': {type(sample)}"
                )
            channels[key].append(value)
    
    return {key: np.array(values, dtype=np.float64) for key, values in channels.items()}


# ============================================================
# 共通ユーティリティ: 窓分割
# ============================================================

def create_windows(
    data: list[Any] | pd.DataFrame,
    window_size: int,
    stride: int,
) -> Iterator[tuple[int, list[Any] | pd.DataFrame]]:
    """
    データを窓に分割するジェネレータ
    
    Args:
        data: サンプルのリストまたはDataFrame
        window_size: 窓サイズ
        stride: ストライド（窓の移動量）
    
    Yields:
        (窓の開始インデックス, 窓データ)
    """
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        if isinstance(data, pd.DataFrame):
            yield start, data.iloc[start:end]
        else:
            yield start, data[start:end]


# ============================================================
# 共通ユーティリティ: ラベル決定
# ============================================================

def determine_label(
    labels: list[Any] | pd.Series,
    strategy: str = "center",
) -> Any:
    """
    窓内のラベルを決定
    
    Args:
        labels: 窓内のラベルリスト or Series
        strategy: ラベル決定戦略
            - "center": 中央のサンプルのラベル
            - "majority": 多数決
            - "first": 最初のサンプルのラベル
            - "last": 最後のサンプルのラベル
    
    Returns:
        決定されたラベル
    """
    if isinstance(labels, pd.Series):
        labels = labels.tolist()
    
    if strategy == "center":
        return labels[len(labels) // 2]
    elif strategy == "majority":
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]
    elif strategy == "first":
        return labels[0]
    elif strategy == "last":
        return labels[-1]
    else:
        raise ValueError(f"Unknown label strategy: {strategy}")


# ============================================================
# 共通基底クラス: リアルタイム抽出器
# ============================================================

class BaseRealtimeExtractor:
    """
    リアルタイム特徴量抽出器の基底クラス
    
    各特徴量抽出ファイルでこのクラスを継承して
    _compute_features() を実装する。
    """
    
    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        *,
        channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    ):
        """
        Args:
            window_size: 窓サイズ（サンプル数）
            stride: ストライド（Noneの場合はwindow_sizeと同じ）
            channel_keys: チャンネルのキー名
        """
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.channel_keys = channel_keys
        
        self._buffer: deque = deque(maxlen=window_size)
        self._samples_since_last_extract: int = 0
        self._is_first_extraction: bool = True
    
    def _compute_features(self, channels: dict[str, np.ndarray]) -> dict[str, float]:
        """
        特徴量を計算（サブクラスで実装）
        
        Args:
            channels: {"ch1": np.array([...]), ...}
        
        Returns:
            特徴量辞書
        """
        raise NotImplementedError("Subclass must implement _compute_features()")
    
    def push(self, sample: Any) -> dict[str, float] | None:
        """
        新しいサンプルを追加し、条件を満たせば特徴量を抽出
        
        Args:
            sample: 新しいサンプル
        
        Returns:
            特徴量辞書（抽出された場合）、None（それ以外）
        """
        self._buffer.append(sample)
        self._samples_since_last_extract += 1
        
        if len(self._buffer) < self.window_size:
            return None
        
        if self._is_first_extraction:
            self._is_first_extraction = False
            self._samples_since_last_extract = 0
            return self._extract()
        
        if self._samples_since_last_extract >= self.stride:
            self._samples_since_last_extract = 0
            return self._extract()
        
        return None
    
    def _extract(self) -> dict[str, float]:
        """現在のバッファから特徴量を抽出"""
        channels = extract_channels(list(self._buffer), self.channel_keys)
        return self._compute_features(channels)
    
    def push_batch(self, samples: list[Any]) -> list[dict[str, float]]:
        """複数のサンプルを一括追加"""
        results = []
        for sample in samples:
            features = self.push(sample)
            if features is not None:
                results.append(features)
        return results
    
    def reset(self) -> None:
        """バッファをクリア"""
        self._buffer.clear()
        self._samples_since_last_extract = 0
        self._is_first_extraction = True
    
    def get_buffer_size(self) -> int:
        """現在のバッファ内サンプル数"""
        return len(self._buffer)
    
    def is_ready(self) -> bool:
        """窓が埋まっているか"""
        return len(self._buffer) >= self.window_size


# ============================================================
# パターン選択ディスパッチャー
# ============================================================

def _get_pattern_module(pattern: str) -> "ModuleType":
    """
    パターン名からモジュールを取得
    
    Args:
        pattern: パターン名（例: "eog_29dim"）
    
    Returns:
        特徴量抽出モジュール
    
    Raises:
        ValueError: パターンが見つからない場合
    """
    try:
        return import_module(f".patterns.{pattern}", package=__name__)
    except ImportError as e:
        available = list_patterns()
        raise ValueError(
            f"Unknown pattern: '{pattern}'. Available patterns: {available}"
        ) from e


def list_patterns() -> list[str]:
    """
    利用可能な特徴量パターン一覧を取得
    
    Returns:
        パターン名のリスト
    """
    patterns_dir = Path(__file__).parent / "patterns"
    patterns = []
    
    for path in patterns_dir.glob("*.py"):
        name = path.stem
        # __init__.py, _で始まるファイルは除外
        if not name.startswith("_"):
            patterns.append(name)
    
    return sorted(patterns)


def get_feature_columns(pattern: str) -> list[str]:
    """
    指定パターンの特徴量カラム名を取得
    
    Args:
        pattern: パターン名
    
    Returns:
        特徴量カラム名のリスト
    """
    module = _get_pattern_module(pattern)
    return module.get_feature_columns()


def batch_extract_from_csv(
    csv_path: str | Path,
    pattern: str,
    window_size: int,
    stride: int | None = None,
    *,
    channel_columns: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    label_column: str | None = "label",
    label_strategy: str = "center",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    CSVファイルから特徴量を一括抽出（パターン指定）
    
    Args:
        csv_path: 入力CSVファイルパス
        pattern: 特徴量パターン名（例: "eog_29dim"）
        window_size: 窓サイズ
        stride: ストライド
        channel_columns: チャンネル列名
        label_column: ラベル列名
        label_strategy: ラベル決定戦略
        output_path: 出力CSVパス
    
    Returns:
        特徴量DataFrame
    """
    module = _get_pattern_module(pattern)
    return module.batch_extract_from_csv(
        csv_path,
        window_size=window_size,
        stride=stride,
        channel_columns=channel_columns,
        label_column=label_column,
        label_strategy=label_strategy,
        output_path=output_path,
    )


def batch_extract_from_dataframe(
    df: pd.DataFrame,
    pattern: str,
    window_size: int,
    stride: int | None = None,
    *,
    channel_columns: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
    label_column: str | None = None,
    label_strategy: str = "center",
) -> pd.DataFrame:
    """
    DataFrameから特徴量を一括抽出（パターン指定）
    
    Args:
        df: 入力DataFrame
        pattern: 特徴量パターン名
        window_size: 窓サイズ
        stride: ストライド
        channel_columns: チャンネル列名
        label_column: ラベル列名
        label_strategy: ラベル決定戦略
    
    Returns:
        特徴量DataFrame
    """
    module = _get_pattern_module(pattern)
    return module.batch_extract_from_dataframe(
        df,
        window_size=window_size,
        stride=stride,
        channel_columns=channel_columns,
        label_column=label_column,
        label_strategy=label_strategy,
    )


def create_realtime_extractor(
    pattern: str,
    window_size: int,
    stride: int | None = None,
    *,
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
) -> BaseRealtimeExtractor:
    """
    リアルタイム特徴量抽出器を生成（パターン指定）
    
    Args:
        pattern: 特徴量パターン名（例: "eog_29dim"）
        window_size: 窓サイズ
        stride: ストライド
        channel_keys: チャンネルのキー名
    
    Returns:
        RealtimeExtractor インスタンス
    """
    module = _get_pattern_module(pattern)
    return module.RealtimeExtractor(
        window_size=window_size,
        stride=stride,
        channel_keys=channel_keys,
    )


def extract_features(
    samples: list[Any],
    pattern: str,
    *,
    channel_keys: tuple[str, ...] = ("ch1", "ch2", "ch3", "ch4"),
) -> dict[str, float]:
    """
    単一窓から特徴量を抽出（パターン指定）
    
    Args:
        samples: 窓サイズ分のサンプルリスト
        pattern: 特徴量パターン名
        channel_keys: チャンネルのキー名
    
    Returns:
        特徴量辞書
    """
    module = _get_pattern_module(pattern)
    return module.extract_features(samples, channel_keys=channel_keys)
