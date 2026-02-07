"""
TabPFN分類処理モジュール

教師データ(context)をもとに、入力データ(input)を分類し結果を保存する。
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts import feature_extractions as fe
from scripts.feature_extractions import create_windows

# .env ファイル読み込み
load_dotenv()


# ============================================================
# TabPFN バージョン定義
# ============================================================

TABPFN_VERSIONS = [
    {"key": "v2", "display": "TabPFN v2"},
    {"key": "v2.5", "display": "TabPFN v2.5"},
]


# ============================================================
# ユーティリティ
# ============================================================

def _prompt_selection(items: list, prompt: str, display_fn) -> any:
    """リストからユーザーに番号入力で選択させる"""
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


def _list_contexts() -> list[dict]:
    """data/context/ 内の教師データ一覧をリストアップ"""
    context_dir = Path(__file__).parent.parent.parent / "data" / "context"
    
    if not context_dir.exists():
        return []
    
    contexts = []
    
    for d in sorted(context_dir.iterdir()):
        if not d.is_dir():
            continue
        
        csv_files = list(d.glob("context_*.csv"))
        metadata_files = list(d.glob("metadata_*.json"))
        
        if not csv_files or not metadata_files:
            continue
        
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        contexts.append({
            "name": d.name,
            "csv_path": csv_files[0],
            "metadata_path": metadata_files[0],
            "metadata": metadata,
        })
    
    return contexts


def _list_inputs() -> list[dict]:
    """data/input/ 内の入力データ一覧をリストアップ"""
    input_dir = Path(__file__).parent.parent.parent / "data" / "input"
    
    if not input_dir.exists():
        return []
    
    inputs = []
    
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        
        csv_files = list(d.glob("input_*.csv"))
        metadata_files = list(d.glob("metadata_*.json"))
        
        if not csv_files or not metadata_files:
            continue
        
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        inputs.append({
            "name": d.name,
            "csv_path": csv_files[0],
            "metadata_path": metadata_files[0],
            "metadata": metadata,
        })
    
    return inputs


def _load_tabpfn_classifier(version_key: str):
    """
    TabPFNClassifier をロード
    
    HuggingFace のゲート付きモデルにアクセスするため、
    .env の HF_TOKEN を環境変数として使用する。
    
    Args:
        version_key: "v2" or "v2.5"
    
    Returns:
        TabPFNClassifier インスタンス（未学習）
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("環境変数 HF_TOKEN が設定されていません (.env ファイルを確認)")
    
    # huggingface_hub が参照する環境変数にセット
    os.environ["HF_TOKEN"] = hf_token
    
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    
    if version_key == "v2":
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    elif version_key == "v2.5":
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
    else:
        raise ValueError(f"Unknown TabPFN version: {version_key}")


# ============================================================
# メイン処理
# ============================================================

def run() -> None:
    """TabPFN分類処理を実行"""
    
    # ----------------------------------------------------------
    # 1. TabPFN バージョン選択
    # ----------------------------------------------------------
    version = _prompt_selection(
        TABPFN_VERSIONS,
        "=== TabPFN バージョンを選択 ===",
        lambda v: v["display"],
    )
    
    # ----------------------------------------------------------
    # 2. 教師データ(context)を選択
    # ----------------------------------------------------------
    contexts = _list_contexts()
    if not contexts:
        print("教師データが見つかりません (data/context/)")
        return
    
    context = _prompt_selection(
        contexts,
        "=== 教師データを選択 ===",
        lambda c: c["metadata"].get("display_name", c["name"]),
    )
    
    # 教師データのメタデータから特徴量パターンを特定
    feature_pattern_path = context["metadata"]["feature_pattern_path"]
    # "scripts/feature_extractions/patterns/first_sample.py" → "first_sample"
    pattern_name = Path(feature_pattern_path).stem
    window_sec = context["metadata"]["window_sec"]
    
    print(f"\n  特徴量パターン: {pattern_name}")
    print(f"  window_sec: {window_sec}")
    
    # ----------------------------------------------------------
    # 3. 入力データ(input)を選択
    # ----------------------------------------------------------
    inputs = _list_inputs()
    if not inputs:
        print("入力データが見つかりません (data/input/)")
        return
    
    input_data = _prompt_selection(
        inputs,
        "=== 入力データを選択 ===",
        lambda i: i["metadata"].get("display_name", i["name"]),
    )
    
    # ----------------------------------------------------------
    # 4. パラメータ計算
    # ----------------------------------------------------------
    input_sampling_rate = input_data["metadata"]["sampling_rate_hz"]
    window_size = int(window_sec * input_sampling_rate)
    stride = window_size  # 重複なし
    
    print(f"\n--- 設定 ---")
    print(f"  TabPFN: {version['display']}")
    print(f"  教師データ: {context['metadata'].get('display_name', context['name'])}")
    print(f"  入力データ: {input_data['metadata'].get('display_name', input_data['name'])}")
    print(f"  特徴量パターン: {pattern_name}")
    print(f"  サンプリングレート: {input_sampling_rate} Hz")
    print(f"  ウィンドウサイズ: {window_size} サンプル")
    
    # ----------------------------------------------------------
    # 5. 教師データ読み込み
    # ----------------------------------------------------------
    print("\n教師データを読み込み中...")
    context_df = pd.read_csv(context["csv_path"])
    
    X_train = context_df.drop(columns=["label"])
    y_train = context_df["label"]
    
    print(f"  教師データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print(f"  ラベル分布: {dict(y_train.value_counts())}")
    
    # ----------------------------------------------------------
    # 6. 入力データから特徴量抽出
    # ----------------------------------------------------------
    print("\n入力データから特徴量を抽出中...")
    input_df = pd.read_csv(input_data["csv_path"])
    
    if window_size > len(input_df):
        print(f"\nエラー: ウィンドウサイズ({window_size})がデータ長({len(input_df)})を超えています")
        return
    
    # label列を除外して特徴量抽出（labelがあっても分類の参考にしない）
    input_features_df = fe.batch_extract_from_dataframe(
        input_df,
        pattern=pattern_name,
        window_size=window_size,
        stride=stride,
        label_column=None,  # labelは参照しない
    )
    
    # 各窓の中間時間 t_us を計算
    t_us_list = []
    for start, window_df in create_windows(input_df, window_size, stride):
        mid_idx = start + window_size // 2
        t_us_list.append(int(input_df.iloc[mid_idx]["t_us"]))
    
    print(f"  抽出されたサンプル数: {len(input_features_df)}")
    
    # ----------------------------------------------------------
    # 7. TabPFN で分類
    # ----------------------------------------------------------
    print("\nTabPFN で分類中...")
    
    model = _load_tabpfn_classifier(version["key"])
    model.fit(X_train, y_train)
    
    predictions = model.predict(input_features_df)
    
    print(f"  分類完了")
    pred_counts = pd.Series(predictions).value_counts()
    print(f"  分類結果分布:")
    for label, count in sorted(pred_counts.items()):
        print(f"    {label}: {count}")
    
    # ----------------------------------------------------------
    # 8. 結果を保存
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(__file__).parent.parent.parent / "data" / "output" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 分類結果CSV（t_us + label）
    result_df = pd.DataFrame({
        "t_us": t_us_list,
        "label": predictions,
    })
    
    csv_path = save_dir / f"output_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)
    
    print(f"\n--- 結果 ---")
    print(f"  保存先: {csv_path}")
    print(f"  総サンプル数: {len(result_df)}")


if __name__ == "__main__":
    run()
