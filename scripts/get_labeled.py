"""
ラベル付きデータ取得モジュール

UIを表示しながら信号を取得し、ラベルを付与してCSVに保存する。
"""

import tkinter as tk
from tkinter import font as tkfont
import random
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import acquisition
import config


# ラベルに対応する表示（矢印など）
LABEL_DISPLAY = {
    "center": "●",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}


class LabeledDataCollector:
    """ラベル付きデータ収集クラス"""
    
    def __init__(self):
        self.current_label: str = config.LABEL_LIST[0]
        self.transition_count: int = 0
        self.label_sequence: list[str] = []
        self.collected_data: list[dict] = []
        
        self.root: Optional[tk.Tk] = None
        self.label_widget: Optional[tk.Label] = None
        self.info_widget: Optional[tk.Label] = None
        self.running: bool = False
        
        self._generate_label_sequence()
    
    def _generate_label_sequence(self) -> None:
        """
        均等に分布したラベルシーケンスを生成
        LABELED_DATA_COUNT回の遷移で各ラベルが最大限均等に登場する
        """
        labels = config.LABEL_LIST
        count = config.LABELED_DATA_COUNT
        
        # 各ラベルの出現回数を計算（均等に分配）
        base_count = count // len(labels)
        remainder = count % len(labels)
        
        # シーケンスを構築
        sequence = []
        for i, label in enumerate(labels):
            # 余りがある場合は先頭のラベルに1回ずつ追加
            times = base_count + (1 if i < remainder else 0)
            sequence.extend([label] * times)
        
        # シャッフル（ただし連続して同じラベルにならないよう調整）
        random.shuffle(sequence)
        
        # 連続を避けるための調整
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                # 後ろから異なるラベルを探して交換
                for j in range(i + 1, len(sequence)):
                    if sequence[j] != sequence[i]:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                        break
        
        self.label_sequence = sequence
    
    def _setup_ui(self) -> None:
        """UIのセットアップ"""
        self.root = tk.Tk()
        self.root.title("ラベル付きデータ取得")
        self.root.geometry("400x300")
        self.root.configure(bg="black")
        
        # 閉じるボタンで停止
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 矢印表示用ラベル
        arrow_font = tkfont.Font(family="Helvetica", size=120)
        self.label_widget = tk.Label(
            self.root,
            text=LABEL_DISPLAY.get(self.current_label, "?"),
            font=arrow_font,
            fg="white",
            bg="black"
        )
        self.label_widget.pack(expand=True)
        
        # 情報表示
        info_font = tkfont.Font(family="Helvetica", size=14)
        self.info_widget = tk.Label(
            self.root,
            text=self._get_info_text(),
            font=info_font,
            fg="gray",
            bg="black"
        )
        self.info_widget.pack(pady=10)
    
    def _get_info_text(self) -> str:
        """情報テキストを生成"""
        return f"{self.current_label} | {self.transition_count + 1}/{config.LABELED_DATA_COUNT}"
    
    def _update_ui(self) -> None:
        """UIを更新"""
        if self.label_widget:
            self.label_widget.config(text=LABEL_DISPLAY.get(self.current_label, "?"))
        if self.info_widget:
            self.info_widget.config(text=self._get_info_text())
    
    def _collect_data(self) -> None:
        """現在キューにあるデータを収集"""
        samples = acquisition.get_data_batch()
        for sample in samples:
            self.collected_data.append({
                "t_us": sample.t_us,
                "ch1": sample.ch1,
                "ch2": sample.ch2,
                "ch3": sample.ch3,
                "ch4": sample.ch4,
                "pc_time": sample.pc_time,
                "label": self.current_label
            })
    
    def _transition(self) -> None:
        """ラベルを遷移"""
        if not self.running:
            return
        
        # データ収集
        self._collect_data()
        
        self.transition_count += 1
        
        # 終了判定
        if self.transition_count >= config.LABELED_DATA_COUNT:
            self._finish()
            return
        
        # 次のラベルに遷移
        self.current_label = self.label_sequence[self.transition_count]
        self._update_ui()
        
        # 次の遷移をスケジュール
        if self.root:
            self.root.after(int(config.LABEL_INTERVAL_SEC * 1000), self._transition)
    
    def _data_collection_loop(self) -> None:
        """定期的にデータを収集"""
        if not self.running:
            return
        
        self._collect_data()
        
        # 100msごとにデータ収集
        if self.root:
            self.root.after(100, self._data_collection_loop)
    
    def _save_csv(self) -> str:
        """CSVファイルを保存"""
        # 保存先ディレクトリ
        save_dir = Path(__file__).parent.parent / "data" / "labeled"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名（タイムスタンプ）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = save_dir / f"labeled_{timestamp}.csv"
        
        # CSV書き込み
        if self.collected_data:
            fieldnames = ["t_us", "ch1", "ch2", "ch3", "ch4", "pc_time", "label"]
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.collected_data)
        
        return str(filepath)
    
    def _finish(self) -> None:
        """処理を終了"""
        self.running = False
        
        # 最後のデータを収集
        self._collect_data()
        
        # 信号取得を停止
        acquisition.stop()
        
        # CSV保存
        filepath = self._save_csv()
        print(f"データを保存しました: {filepath}")
        print(f"総サンプル数: {len(self.collected_data)}")
        
        # UIを閉じる
        if self.root:
            self.root.destroy()
    
    def _on_close(self) -> None:
        """ウィンドウが閉じられた時"""
        self._finish()
    
    def run(self) -> None:
        """メイン処理を実行"""
        # 信号取得を開始
        acquisition.start(
            port=config.PORT,
            baudrate=config.BAUDRATE,
            adc_resolution=config.ADC_RESOLUTION
        )
        
        # キューをクリア
        acquisition.clear_queue()
        
        # UIセットアップ
        self._setup_ui()
        
        self.running = True
        
        # 初期ラベル設定
        self.current_label = self.label_sequence[0] if self.label_sequence else config.LABEL_LIST[0]
        self._update_ui()
        
        # データ収集ループ開始
        if self.root:
            self.root.after(100, self._data_collection_loop)
        
        # 最初の遷移をスケジュール
        if self.root:
            self.root.after(int(config.LABEL_INTERVAL_SEC * 1000), self._transition)
        
        # UIメインループ
        if self.root:
            self.root.mainloop()


def run() -> None:
    """ラベル付きデータ取得を実行"""
    collector = LabeledDataCollector()
    collector.run()


if __name__ == "__main__":
    run()
