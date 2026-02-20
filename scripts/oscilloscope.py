"""
オシロスコープモジュール

シリアルポートから4ch ADC信号をリアルタイムに取得し、
tkinterキャンバス上に移動プロットする。
"""

import tkinter as tk
from collections import deque
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import acquisition
import config


# ============================================================
# パラメータ
# ============================================================

# プロット領域のサイズ
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 400

# Y軸のスケール（ADC値の範囲）
Y_MIN = -1000
Y_MAX = 5000

# プロット時間幅（秒）
TIME_WINDOW_SEC = 5.0

# 更新間隔（ミリ秒）
UPDATE_INTERVAL_MS = 50

# チャンネルごとの色
CHANNEL_COLORS = {
    "ch1": "#FF0000",  # 赤
    "ch2": "#00FF00",  # 緑
    "ch3": "#0000FF",  # 青
    "ch4": "#FFFF00",  # 黄
}


# ============================================================
# オシロスコープクラス
# ============================================================

class Oscilloscope:
    """リアルタイムオシロスコープ"""

    def __init__(self):
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.running: bool = False

        # サンプリングレートから最大サンプル数を計算
        self.max_samples = int(config.SAMPLING_RATE_HZ * TIME_WINDOW_SEC)

        # 各チャンネルのデータバッファ（FIFO）
        self.buffers = {
            "ch1": deque(maxlen=self.max_samples),
            "ch2": deque(maxlen=self.max_samples),
            "ch3": deque(maxlen=self.max_samples),
            "ch4": deque(maxlen=self.max_samples),
        }

    def _setup_ui(self) -> None:
        """UIのセットアップ"""
        self.root = tk.Tk()
        self.root.title("オシロスコープ - 4ch ADC")
        self.root.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT + 100}")
        self.root.configure(bg="black")

        # 閉じるボタンで停止
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 情報表示ラベル
        info_text = (
            f"4ch ADC オシロスコープ | "
            f"サンプリングレート: {config.SAMPLING_RATE_HZ} Hz | "
            f"時間幅: {TIME_WINDOW_SEC} 秒 | "
            f"Y軸: {Y_MIN} ~ {Y_MAX}"
        )
        info_label = tk.Label(
            self.root,
            text=info_text,
            bg="black",
            fg="white",
            font=("Courier", 10),
        )
        info_label.pack(side=tk.TOP, pady=5)

        # チャンネル凡例
        legend_frame = tk.Frame(self.root, bg="black")
        legend_frame.pack(side=tk.TOP, pady=5)

        for ch_name, color in CHANNEL_COLORS.items():
            label = tk.Label(
                legend_frame,
                text=f"● {ch_name.upper()}",
                bg="black",
                fg=color,
                font=("Courier", 12, "bold"),
            )
            label.pack(side=tk.LEFT, padx=10)

        # キャンバス（プロット領域）
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP)

        # グリッド線を描画
        self._draw_grid()

    def _draw_grid(self) -> None:
        """グリッド線を描画"""
        if not self.canvas:
            return

        # 水平グリッド（Y軸の目盛り）
        grid_color = "#333333"
        num_h_lines = 5
        for i in range(num_h_lines + 1):
            y = CANVAS_HEIGHT * i / num_h_lines
            self.canvas.create_line(
                0, y, CANVAS_WIDTH, y, fill=grid_color, dash=(2, 4)
            )
            # Y軸の値ラベル
            value = Y_MAX - (Y_MAX - Y_MIN) * i / num_h_lines
            self.canvas.create_text(
                10, y, text=f"{int(value)}", fill="gray", anchor="nw", font=("Courier", 8)
            )

        # 垂直グリッド（時間軸の目盛り）
        num_v_lines = 10
        for i in range(num_v_lines + 1):
            x = CANVAS_WIDTH * i / num_v_lines
            self.canvas.create_line(
                x, 0, x, CANVAS_HEIGHT, fill=grid_color, dash=(2, 4)
            )

    def _adc_to_canvas_y(self, adc_value: float) -> float:
        """ADC値をキャンバスY座標に変換"""
        # Y軸は上が0、下がCANVAS_HEIGHT（反転）
        normalized = (adc_value - Y_MIN) / (Y_MAX - Y_MIN)
        normalized = max(0.0, min(1.0, normalized))  # クリップ
        return CANVAS_HEIGHT * (1.0 - normalized)

    def _update_plot(self) -> None:
        """プロットを更新"""
        if not self.running or not self.canvas:
            return

        # 新しいサンプルを取得
        samples = acquisition.get_data_batch()
        for sample in samples:
            self.buffers["ch1"].append(sample.ch1)
            self.buffers["ch2"].append(sample.ch2)
            self.buffers["ch3"].append(sample.ch3)
            self.buffers["ch4"].append(sample.ch4)

        # キャンバスをクリア（グリッドは残す）
        self.canvas.delete("plot")

        # 各チャンネルをプロット
        for ch_name, color in CHANNEL_COLORS.items():
            data = list(self.buffers[ch_name])
            if len(data) < 2:
                continue

            # X座標は左端=最古、右端=最新
            points = []
            for i, value in enumerate(data):
                x = CANVAS_WIDTH * i / max(len(data) - 1, 1)
                y = self._adc_to_canvas_y(value)
                points.extend([x, y])

            # 折れ線グラフを描画
            if len(points) >= 4:  # 最低2点必要
                self.canvas.create_line(
                    *points, fill=color, width=2, tags="plot"
                )

        # 次回の更新をスケジュール
        if self.root:
            self.root.after(UPDATE_INTERVAL_MS, self._update_plot)

    def _on_close(self) -> None:
        """ウィンドウが閉じられた時"""
        self._finish()

    def _finish(self) -> None:
        """処理を終了"""
        self.running = False

        # 信号取得が開始済みの場合のみ停止
        if acquisition.is_running():
            acquisition.stop()

        # UIを閉じる
        if self.root:
            self.root.destroy()

    def run(self) -> None:
        """メイン処理を実行"""
        print("\n=== オシロスコープ起動 ===")
        print(f"サンプリングレート: {config.SAMPLING_RATE_HZ} Hz")
        print(f"時間幅: {TIME_WINDOW_SEC} 秒")
        print(f"Y軸範囲: {Y_MIN} ~ {Y_MAX}")
        print("ウィンドウを閉じると終了します")
        print("=" * 40)

        # 信号取得を開始
        acquisition.start(
            port=config.PORT,
            baudrate=config.BAUDRATE,
        )

        # キュークリアは行わない（他のプロセスのデータを消さないため）
        # オシロスコープは表示用途なので、現時点以降のデータのみ使用

        # UIセットアップ
        self._setup_ui()

        self.running = True

        # プロット更新開始
        if self.root:
            self.root.after(UPDATE_INTERVAL_MS, self._update_plot)

        # UIメインループ
        if self.root:
            self.root.mainloop()


# ============================================================
# エントリーポイント
# ============================================================

def run() -> None:
    """オシロスコープを起動"""
    oscilloscope = Oscilloscope()
    oscilloscope.run()


if __name__ == "__main__":
    run()
