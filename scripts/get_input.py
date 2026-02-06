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
        self.lock = threading.Lock()
        
        # 現在押下中のキー
        self.pressed_keys: set = set()
        
        # キーボードリスナー
        self.listener: Optional[keyboard.Listener] = None
    
    def _on_key_press(self, key) -> None:
        """キー押下時の処理"""
        try:
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
            
            # xキーで終了
            if hasattr(key, 'char') and key.char == 'x':
                self._finish()
        except AttributeError:
            pass
    
    def _on_key_release(self, key) -> None:
        """キー解放時の処理"""
        try:
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
                    "pc_time": sample.pc_time,
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
        
        # CSV書き込み
        if self.collected_data:
            fieldnames = ["t_us", "ch1", "ch2", "ch3", "ch4", "pc_time", "label"]
            with open(csv_filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.collected_data)
        
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
        
        # キーボードリスナーを停止
        if self.listener:
            self.listener.stop()
        
        # 信号取得を停止
        acquisition.stop()
        
        # CSV保存
        filepath = self._save_csv()
        print(f"データを保存しました: {filepath}")
        print(f"総サンプル数: {len(self.collected_data)}")
    
    def run(self) -> None:
        """メイン処理を実行"""
        print("\n=== 入力データ取得 ===")
        print("十字キーを押下して入力をラベル付けしてください")
        print("  左キー → left")
        print("  右キー → right")
        print("  上キー → up")
        print("  下キー → down")
        print("  未押下 → center")
        print("\nxキーを押すと終了します")
        print("=" * 40)
        
        # 信号取得を開始
        acquisition.start(
            port=config.PORT,
            baudrate=config.BAUDRATE,
        )
        
        # キューをクリア
        acquisition.clear_queue()
        
        self.running = True
        
        # データ収集スレッドを開始
        collection_thread = threading.Thread(target=self._collect_data_loop, daemon=True)
        collection_thread.start()
        
        # キーボードリスナーを開始
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        self.listener.join()
        
        # 終了後、データ収集スレッドの終了を待つ
        collection_thread.join(timeout=1.0)


def run() -> None:
    """入力データ取得を実行"""
    collector = InputDataCollector()
    collector.run()


if __name__ == "__main__":
    run()
