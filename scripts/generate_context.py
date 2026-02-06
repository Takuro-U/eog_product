"""
教師データ（コンテキスト）生成モジュール

ラベル付き生信号と特徴量パターンを選択し、
TabPFN用の教師データCSVとメタデータを生成する。
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from scripts import feature_extractions as fe
from scripts.feature_extractions import create_windows

# .env ファイル読み込み
load_dotenv()


# ============================================================
# ユーティリティ
# ============================================================

def _list_labeled_datasets() -> list[dict]:
    """
    data/labeled/ 内のラベル付き信号一覧をリストアップ
    
    Returns:
        データセット情報の辞書リスト
    """
    labeled_dir = Path(__file__).parent.parent / "data" / "labeled"
    
    if not labeled_dir.exists():
        return []
    
    datasets = []
    
    for d in sorted(labeled_dir.iterdir()):
        if not d.is_dir():
            continue
        
        csv_files = list(d.glob("labeled_*.csv"))
        metadata_files = list(d.glob("metadata_*.json"))
        
        if not csv_files or not metadata_files:
            continue
        
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        datasets.append({
            "name": d.name,
            "csv_path": csv_files[0],
            "metadata_path": metadata_files[0],
            "metadata": metadata,
        })
    
    return datasets


def _prompt_selection(items: list, prompt: str, display_fn) -> any:
    """
    リストからユーザーに番号入力で選択させる
    
    Args:
        items: 選択肢のリスト
        prompt: 表示するプロンプト
        display_fn: 各アイテムの表示文字列を返す関数
    
    Returns:
        選択されたアイテム
    """
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"  {i + 1}: {display_fn(item)}")
    
    while True:
        try:
            choice = input("番号を入力 > ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print(f"1〜{len(items)} の範囲で入力してください")
        except ValueError:
            print("数値を入力してください")


def _determine_label(labels: pd.Series) -> str:
    """
    窓内のラベルを決定
    
    - 全サンプルが同一ラベル → そのラベル
    - 複数ラベルが混在（ラベル跨ぎ） → "transition"
    
    Args:
        labels: 窓内のラベル Series
    
    Returns:
        決定されたラベル文字列
    """
    unique = labels.unique()
    if len(unique) == 1:
        return str(unique[0])
    return "transition"


# ============================================================
# メイン処理
# ============================================================

def run() -> None:
    """教師データ生成を実行"""
    
    # ----------------------------------------------------------
    # 1. ラベル付き信号を選択
    # ----------------------------------------------------------
    datasets = _list_labeled_datasets()
    if not datasets:
        print("ラベル付き信号が見つかりません (data/labeled/)")
        return
    
    dataset = _prompt_selection(
        datasets,
        "=== ラベル付き信号を選択 ===",
        lambda d: f"{d['name']}  (sampling_rate: {d['metadata']['sampling_rate_hz']} Hz)",
    )
    
    # ----------------------------------------------------------
    # 2. 特徴量パターンを選択
    # ----------------------------------------------------------
    patterns = fe.list_patterns()
    if not patterns:
        print("特徴量パターンが見つかりません (scripts/feature_extractions/patterns/)")
        return
    
    pattern = _prompt_selection(
        patterns,
        "=== 特徴量パターンを選択 ===",
        lambda p: p,
    )
    
    # ----------------------------------------------------------
    # 3. パラメータ計算
    # ----------------------------------------------------------
    sampling_rate_hz = dataset["metadata"]["sampling_rate_hz"]
    window_size = int(config.WINDOW_SEC * sampling_rate_hz)
    stride = window_size  # 重複なし（教師データの独立性を確保）
    
    print(f"\n--- 設定 ---")
    print(f"  ラベル付き信号: {dataset['name']}")
    print(f"  特徴量パターン: {pattern}")
    print(f"  WINDOW_SEC: {config.WINDOW_SEC}")
    print(f"  サンプリングレート: {sampling_rate_hz} Hz")
    print(f"  ウィンドウサイズ: {window_size} サンプル")
    
    # ----------------------------------------------------------
    # 4. データ読み込み
    # ----------------------------------------------------------
    df = pd.read_csv(dataset["csv_path"])
    print(f"  総サンプル数: {len(df)}")
    
    if window_size > len(df):
        print(f"\nエラー: ウィンドウサイズ({window_size})がデータ長({len(df)})を超えています")
        return
    
    # ----------------------------------------------------------
    # 5. 特徴量抽出（ラベルなし）
    # ----------------------------------------------------------
    features_df = fe.batch_extract_from_dataframe(
        df,
        pattern=pattern,
        window_size=window_size,
        stride=stride,
        label_column=None,
    )
    
    # ----------------------------------------------------------
    # 6. ラベル付与（跨ぎ判定付き）
    # ----------------------------------------------------------
    labels = []
    for _, window_df in create_windows(df, window_size, stride):
        labels.append(_determine_label(window_df["label"]))
    
    features_df["label"] = labels
    
    # ----------------------------------------------------------
    # 7. 保存
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(__file__).parent.parent / "data" / "context" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 教師データCSV
    csv_path = save_dir / f"context_{timestamp}.csv"
    features_df.to_csv(csv_path, index=False)
    
    # メタデータJSON（パスはプロジェクトルート起点の相対パス）
    project_root_str = os.getenv("PROJECT_ROOT")
    if not project_root_str:
        raise RuntimeError("環境変数 PROJECT_ROOT が設定されていません (.env ファイルを確認)")
    
    project_root = Path(project_root_str)
    metadata = {
        "labeled_signal_path": str(dataset["csv_path"].relative_to(project_root)),
        "feature_pattern_path": str(
            (Path("scripts") / "feature_extractions" / "patterns" / f"{pattern}.py")
        ),
        "window_sec": config.WINDOW_SEC,
    }
    metadata_path = save_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # ----------------------------------------------------------
    # 8. 結果表示
    # ----------------------------------------------------------
    print(f"\n--- 結果 ---")
    print(f"  生成サンプル数: {len(features_df)}")
    print(f"  特徴量次元数: {len(features_df.columns) - 1}")  # label列を除く
    
    label_counts = features_df["label"].value_counts()
    print(f"  ラベル分布:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    
    print(f"\n  保存先: {save_dir}")
    print(f"    {csv_path.name}")
    print(f"    {metadata_path.name}")


if __name__ == "__main__":
    run()
