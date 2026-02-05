# get_labeled.py - 実装詳細

## 概要

UIを表示しながらシリアルからADC信号を取得し、その瞬間のラベル（視線方向）を付与してCSVに保存するモジュール。機械学習用の教師データ作成を目的とする。

## 設計要件

1. **信号取得とUI表示の同時実行**: `acquisition`モジュールで信号取得、tkinterでUI表示
2. **内部状態としてのラベル**: `LABEL_LIST`の要素を保持、初期値は0番目
3. **定期的な遷移**: `LABEL_INTERVAL_SEC`ごとに異なるラベルに遷移
4. **均等な分布**: `LABELED_DATA_COUNT`回の遷移内で各ラベルが最大限均等に登場
5. **リアルタイムUI**: 矢印などでラベルを視覚的に表示
6. **CSV出力**: label列を追加した信号データを保存
7. **自動終了**: 指定回数の遷移後に処理終了

## 主要クラス

### LabeledDataCollector

```python
class LabeledDataCollector:
    current_label: str       # 現在のラベル
    transition_count: int    # 遷移回数
    label_sequence: list     # 事前生成されたラベルシーケンス
    collected_data: list     # 収集したデータ
```

## ラベルシーケンス生成アルゴリズム

`_generate_label_sequence()`で以下の処理を行う：

1. 各ラベルの出現回数を均等に分配（`LABELED_DATA_COUNT // len(LABEL_LIST)`）
2. 余りは先頭のラベルから順に1回ずつ追加
3. シャッフル
4. 連続して同じラベルにならないよう調整

例: `LABELED_DATA_COUNT=15`, `LABEL_LIST`が5要素の場合
→ 各ラベルが3回ずつ登場

## データフロー

```
[acquisition] → get_data_batch() → collected_data[] → CSV
                    ↓
              現在のcurrent_labelを付与
```

## タイマー構成

- **データ収集ループ**: 100msごとに`acquisition.get_data_batch()`を呼び出し
- **遷移タイマー**: `LABEL_INTERVAL_SEC`ごとにラベルを遷移

## UI構成

- tkinter使用
- 黒背景に白文字で矢印を大きく表示
- 下部に現在のラベル名と進捗を表示

### ラベル表示マッピング

```python
LABEL_DISPLAY = {
    "center": "●",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}
```

## 出力

### 保存先
`/data/labeled/labeled_YYYYMMDD_HHMMSS.csv`

### CSV形式
```csv
t_us,ch1,ch2,ch3,ch4,pc_time,label
123456,2048,2050,2045,2047,1234567890.123,center
```

## 依存関係

- `acquisition`: 信号取得
- `config`: 設定値（PORT, BAUDRATE, ADC_RESOLUTION, LABEL_*）
- `tkinter`: UI表示（Python標準ライブラリ）

## 終了条件

1. `LABELED_DATA_COUNT`回目の遷移時
2. ウィンドウの×ボタンが押された時

いずれの場合も：
- 残りのデータを収集
- `acquisition.stop()`を呼び出し
- CSVを保存
- UIを閉じる

## 改修時の注意

- `root.after()`でタイマーを管理しているため、UIスレッドと同期的に動作
- データ収集は100msごとだが、250Hzサンプリングなら十分（4ms間隔で約25サンプル/100ms）
- ラベル遷移時ではなく、各サンプル取得時点のラベルが付与される
