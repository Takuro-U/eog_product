# シリアル通信設定
PORT = "/dev/cu.usbmodem103"
BAUDRATE = 115200

# ADC設定
SAMPLING_RATE_HZ = 250
ADC_RESOLUTION = 12  # ビット数

# ラベル付きデータ取得設定
LABEL_INTERVAL_SEC = 2
LABEL_LIST = [
    "center",
    "left",
    "right",
    "up",
    "down",
]
