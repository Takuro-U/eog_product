# シリアル通信設定
# PORT = "/dev/cu.usbmodem103"
PORT = "/dev/cu.usbmodemF0F5BD5281842"
BAUDRATE = 115200

# ADC設定
SAMPLING_RATE_HZ = 250
ADC_RESOLUTION = 12  # ビット数

# ラベル付きデータ取得設定
LABEL_INTERVAL_SEC = 2
LABELED_DATA_COUNT = 15 # LABEL_LISTの倍数推奨
LABEL_LIST = [
    "center", # これは固定
    "left",
    "right",
    "up",
    "down",
]
