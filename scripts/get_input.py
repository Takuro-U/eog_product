"""
入力データ取得モジュール

リアルタイムで信号を取得し、PCの十字キー入力をラベルとして記録する。
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import threading

from pynput import keyboard

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import acquisition
import config


class InputDataCollector:
    """入力データ収集クラス"""

    def __init__(self):
        self.collected_data: list[dict] = []
        self.current_label: str = config.LABEL_LIST[0]  # デフォルトは"center"
        self.running: bool = False
        self.measuring: bool = False  # 測定中フラグ（待機中=False, 測定中=True）
        self.lock = threading.Lock()

        # 現在押下中のキー
        self.pressed_keys: set = set()

        # キーボードリスナー
        self.listener: Optional[keyboard.Listener] = None
        # データ収集スレッド
        self.collection_thread: Optional[threading.Thread] = None
    
    def _on_key_press(self, key) -> None:
        """キー押下時の処理"""
        try:
            # スペースキーの判定（測定開始/終了）
            if key == keyboard.Key.space:
                if not self.measuring:
                    self._start_measurement()
                else:
                    self._finish()
                return

            # 測定中のみ十字キー入力を受け付ける
            if not self.measuring:
                return

            # 十字キーの判定
            if key == keyboard.Key.left:
                with self.lock:
                    self.pressed_keys.add('left')
                    self._update_label()
            elif key == keyboard.Key.right:
                with self.lock:
                    self.pressed_keys.add('right')
                    self._update_label()
            elif key == keyboard.Key.up:
                with self.lock:
                    self.pressed_keys.add('up')
                    self._update_label()
            elif key == keyboard.Key.down:
                with self.lock:
                    self.pressed_keys.add('down')
                    self._update_label()
        except AttributeError:
            pass
    
    def _on_key_release(self, key) -> None:
        """キー解放時の処理"""
        try:
            # 測定中のみ十字キー解放を受け付ける
            if not self.measuring:
                return

            # 十字キーの判定
            if key == keyboard.Key.left:
                with self.lock:
                    self.pressed_keys.discard('left')
                    self._update_label()
            elif key == keyboard.Key.right:
                with self.lock:
                    self.pressed_keys.discard('right')
                    self._update_label()
            elif key == keyboard.Key.up:
                with self.lock:
                    self.pressed_keys.discard('up')
                    self._update_label()
            elif key == keyboard.Key.down:
                with self.lock:
                    self.pressed_keys.discard('down')
                    self._update_label()
        except AttributeError:
            pass
    
    def _update_label(self) -> None:
        """
        押下中のキーに応じてラベルを更新
        優先順位: 左 > 右 > 上 > 下 > center
        """
        if 'left' in self.pressed_keys:
            self.current_label = config.LABEL_LIST[1]  # left
        elif 'right' in self.pressed_keys:
            self.current_label = config.LABEL_LIST[2]  # right
        elif 'up' in self.pressed_keys:
            self.current_label = config.LABEL_LIST[3]  # up
        elif 'down' in self.pressed_keys:
            self.current_label = config.LABEL_LIST[4]  # down
        else:
            self.current_label = config.LABEL_LIST[0]  # center

    def _start_measurement(self) -> None:
        """スペースキー押下で測定を開始"""
        if self.measuring:
            return

        print("\n測定中")
        print("スペースキーで終了")

        # シリアル通信開始・キュークリア
        acquisition.start(
            port=config.PORT,
            baudrate=config.BAUDRATE,
        )
        acquisition.clear_queue()

        self.running = True
        self.measuring = True

        # データ収集スレッドを開始
        self.collection_thread = threading.Thread(target=self._collect_data_loop, daemon=True)
        self.collection_thread.start()
    
    def _collect_data_loop(self) -> None:
        """データ収集ループ"""
        while self.running:
            # キューからデータを取得
            samples = acquisition.get_data_batch()
            
            for sample in samples:
                with self.lock:
                    current_label = self.current_label
                
                self.collected_data.append({
                    "t_us": sample.t_us,
                    "ch1": sample.ch1,
                    "ch2": sample.ch2,
                    "ch3": sample.ch3,
                    "ch4": sample.ch4,
                    "label": current_label
                })
            
            # 少し待機（CPU使用率を抑える）
            import time
            time.sleep(0.01)
    
    def _save_csv(self) -> str:
        """CSVファイルとメタデータを保存"""
        # タイムスタンプ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存先ディレクトリ
        save_dir = Path(__file__).parent.parent / "data" / "input" / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVファイルパス
        csv_filepath = save_dir / f"input_{timestamp}.csv"
        
        # CSV書き込み（t_usを取得開始からの経過時間に変換）
        if self.collected_data:
            # 最初のサンプルのt_usを基準とする
            t_us_offset = self.collected_data[0]["t_us"]
            
            # 経過時間に変換したデータを作成
            converted_data = []
            for row in self.collected_data:
                converted_row = row.copy()
                converted_row["t_us"] = row["t_us"] - t_us_offset
                converted_data.append(converted_row)
            
            fieldnames = ["t_us", "ch1", "ch2", "ch3", "ch4", "label"]
            with open(csv_filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(converted_data)
        
        # メタデータ保存
        metadata = {
            "display_name": timestamp,
            "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        }
        metadata_filepath = save_dir / f"metadata_{timestamp}.json"
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(csv_filepath)
    
    def _finish(self) -> None:
        """処理を終了"""
        if not self.running:
            return

        print("\n\n終了処理を開始します...")
        self.running = False
        self.measuring = False

        # キーボードリスナーを停止
        if self.listener:
            self.listener.stop()

        # 信号取得が開始済みの場合のみ停止・保存
        if acquisition.is_running():
            acquisition.stop()

            filepath = self._save_csv()
            print(f"データを保存しました: {filepath}")
            print(f"総サンプル数: {len(self.collected_data)}")
    
    def run(self) -> None:
        """メイン処理を実行"""
        print("\n=== 入力データ取得 ===")
        print("スペースキーを押すと測定を開始します")
        print("【操作方法】")
        print("  左キー → left")
        print("  右キー → right")
        print("  上キー → up")
        print("  下キー → down")
        print("  未押下 → center")
        print("スペースキーを押すと測定を終了します")
        print("=" * 40)

        self.running = True

        # キーボードリスナーを開始（待機状態）
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        self.listener.join()

        # 終了後、データ収集スレッドの終了を待つ
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)


def run() -> None:
    """入力データ取得を実行"""
    collector = InputDataCollector()
    collector.run()


if __name__ == "__main__":
    run()
