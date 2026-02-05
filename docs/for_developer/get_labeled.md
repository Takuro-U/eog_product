# get_labeled - ラベル付きデータ取得

UIを表示しながら信号を取得し、ラベルを付与してCSVに保存する。

## 使い方

```python
from scripts import get_labeled

get_labeled.run()
```

または直接実行:

```bash
python scripts/get_labeled.py
```

## 動作

1. UIが表示され、矢印（←→↑↓●）で現在のラベルを表示
2. `LABEL_INTERVAL_SEC`秒ごとにラベルが変化
3. `LABELED_DATA_COUNT`回の遷移後、自動終了
4. `/data/labeled/labeled_YYYYMMDD_HHMMSS.csv`に保存

## 設定（config.py）

```python
LABEL_INTERVAL_SEC = 2    # ラベル遷移間隔（秒）
LABELED_DATA_COUNT = 15   # 総遷移回数
LABEL_LIST = ["center", "left", "right", "up", "down"]
```

## 出力CSV形式

```csv
t_us,ch1,ch2,ch3,ch4,pc_time,label
123456,2048,2050,2045,2047,1234567890.123,center
```
