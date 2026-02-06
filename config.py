# シリアル通信設定
# PORT = "/dev/cu.usbmodem1103"
PORT = "/dev/cu.usbmodemF0F5BD5281842"
BAUDRATE = 115200

# 以下はCSV出力時の設定
# CSV固有の値はメタデータ用jsonを参照
SAMPLING_RATE_HZ = 250 # **測定時の** サンプリングレート
WINDOW_SEC = 0.1 # **教師データ生成時の** ウィンドウサイズ(秒)

# ラベル付きデータ取得設定
LABEL_INTERVAL_SEC = 2
LABELED_DATA_COUNT = 15 # LABEL_LISTの倍数推奨
LABEL_LIST = [ # 基本的に固定 ラベル付きデータを取得する際コメントアウトで除外可能
    "center",
    "left",
    "right",
    "up",
    "down",
]
