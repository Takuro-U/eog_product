# acquisition - シリアル信号取得

マイコンから ADC 信号をリアルタイムに取得するモジュール。

## 基本的な使い方

```python
from scripts import acquisition
import config

# 起動時に一度だけ
acquisition.start(
    port=config.PORT,
    baudrate=config.BAUDRATE,
    adc_resolution=config.ADC_RESOLUTION
)

# 最新値を取得
sample = acquisition.get_latest()
if sample:
    print(f"CH1={sample.ch1}, CH2={sample.ch2}, CH3={sample.ch3}, CH4={sample.ch4}")

# 終了時
acquisition.stop()
```

## データ形式

```python
sample.t_us    # マイコン側タイムスタンプ (μs)
sample.ch1     # チャンネル1の値
sample.ch2     # チャンネル2の値
sample.ch3     # チャンネル3の値
sample.ch4     # チャンネル4の値
sample.pc_time # PC側受信時刻
```

## 主要 API

| 関数                                    | 説明                     |
| --------------------------------------- | ------------------------ |
| `start(port, baudrate, adc_resolution)` | 取得開始                 |
| `stop()`                                | 取得停止                 |
| `get_latest()`                          | 最新値を取得             |
| `get_data(timeout)`                     | キューから取得（待機可） |
| `get_data_batch()`                      | 一括取得                 |

## get_latest vs get_data

- **get_latest()**: 最新値だけ欲しい場合（同じ値を複数回取得する可能性あり）
- **get_data()**: 順番に処理したい場合（重複なし）

## 注意

- `start()`は一度だけ呼ぶ
- データを漏れなく処理したい場合は`get_data()`を使用
