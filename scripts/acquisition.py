"""
シリアルポートからADC信号をリアルタイムに取得するモジュール

マイコンからの送信形式:
    - ヘッダ行: "t_us,ch1,ch2,ch3,ch4"
    - データ行: "timestamp_us,ch1_value,ch2_value,ch3_value,ch4_value"
    - 4チャンネル (ch1, ch2, ch3, ch4)

使用例:
    from scripts import acquisition
    import config
    
    # 取得開始（呼び出し側で設定値を渡す）
    acquisition.start(
        port=config.PORT,
        baudrate=config.BAUDRATE,
        adc_resolution=config.ADC_RESOLUTION
    )
    
    # 任意の箇所から最新値を取得
    sample = acquisition.get_latest()
    if sample:
        print(f"CH1={sample.ch1}, CH2={sample.ch2}, CH3={sample.ch3}, CH4={sample.ch4}")
    
    # 終了時
    acquisition.stop()
"""

import threading
import queue
import time
from typing import Optional, NamedTuple

import serial


class AdcSample(NamedTuple):
    """ADCサンプルデータ"""
    t_us: int       # マイコン側のタイムスタンプ (マイクロ秒)
    ch1: int        # チャンネル1の値
    ch2: int        # チャンネル2の値
    ch3: int        # チャンネル3の値
    ch4: int        # チャンネル4の値


# ============================================================
# モジュール内部状態（シングルトン）
# ============================================================

_serial: Optional[serial.Serial] = None
_thread: Optional[threading.Thread] = None
_running = threading.Event()
_lock = threading.Lock()

_latest_sample: Optional[AdcSample] = None
_data_queue: queue.Queue = queue.Queue(maxsize=1000)

# 設定値（start時に設定される）
_adc_resolution: int = 12
_adc_max_value: int = 4095


# ============================================================
# 内部関数
# ============================================================

def _parse_line(raw_line: bytes) -> Optional[AdcSample]:
    """受信行をパースしてAdcSampleを返す"""
    try:
        line = raw_line.decode('utf-8').strip()
        
        if not line:
            return None
        
        # ヘッダー行をスキップ
        if line == "t_us,ch1,ch2,ch3,ch4":
            return None
        
        # CSV形式をパース
        parts = line.split(',')
        if len(parts) != 5:
            return None
        
        t_us = int(parts[0])
        ch1 = int(parts[1])
        ch2 = int(parts[2])
        ch3 = int(parts[3])
        ch4 = int(parts[4])
        
        return AdcSample(t_us=t_us, ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4)
    
    except (ValueError, UnicodeDecodeError):
        return None


def _read_loop() -> None:
    """バックグラウンドでシリアルデータを読み取るループ"""
    global _latest_sample
    
    while _running.is_set():
        try:
            if _serial is None or not _serial.is_open:
                break
            
            raw_line = _serial.readline()
            if not raw_line:
                continue
            
            sample = _parse_line(raw_line)
            if sample is None:
                continue
            
            # 最新値を更新
            with _lock:
                _latest_sample = sample
            
            # キューに追加（満杯の場合は古いデータを破棄）
            try:
                _data_queue.put_nowait(sample)
            except queue.Full:
                try:
                    _data_queue.get_nowait()
                    _data_queue.put_nowait(sample)
                except queue.Empty:
                    pass
        
        except serial.SerialException as e:
            print(f"シリアル通信エラー: {e}")
            break
        except Exception as e:
            print(f"予期しないエラー: {e}")
            continue


# ============================================================
# 公開API
# ============================================================

def start(
    port: str,
    baudrate: int,
    adc_resolution: int = 12,
    timeout: float = 0.1
) -> None:
    """
    データ取得を開始
    
    Args:
        port: シリアルポート
        baudrate: ボーレート
        adc_resolution: ADC解像度（ビット数、デフォルト12）
        timeout: シリアル読み取りのタイムアウト（秒）
    """
    global _serial, _thread, _adc_resolution, _adc_max_value
    
    if _running.is_set():
        return
    
    _adc_resolution = adc_resolution
    _adc_max_value = (1 << adc_resolution) - 1
    
    _serial = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=timeout
    )
    
    _running.set()
    _thread = threading.Thread(target=_read_loop, daemon=True)
    _thread.start()


def stop() -> None:
    """データ取得を停止"""
    global _serial, _thread
    
    _running.clear()
    
    if _thread is not None:
        _thread.join(timeout=2.0)
        _thread = None
    
    if _serial is not None:
        _serial.close()
        _serial = None


def get_latest() -> Optional[AdcSample]:
    """
    最新のサンプルを取得
    
    Returns:
        AdcSample または None（データがない場合）
    """
    with _lock:
        return _latest_sample


def get_data(timeout: Optional[float] = None) -> Optional[AdcSample]:
    """
    キューからデータを取得
    
    Args:
        timeout: タイムアウト（秒）。Noneの場合はブロッキングしない
    
    Returns:
        AdcSample または None
    """
    try:
        if timeout is None:
            return _data_queue.get_nowait()
        else:
            return _data_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def get_data_batch(max_count: int = None) -> list[AdcSample]:
    """
    キューから複数のデータを一括取得（ノンブロッキング）
    
    Args:
        max_count: 取得する最大数。Noneの場合は全て取得
    
    Returns:
        AdcSampleのリスト
    """
    samples = []
    count = 0
    while max_count is None or count < max_count:
        try:
            sample = _data_queue.get_nowait()
            samples.append(sample)
            count += 1
        except queue.Empty:
            break
    return samples


def clear_queue() -> int:
    """
    キューをクリア
    
    Returns:
        クリアしたデータ数
    """
    count = 0
    while True:
        try:
            _data_queue.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count


def is_running() -> bool:
    """データ取得が実行中かどうか"""
    return _running.is_set()


def queue_size() -> int:
    """現在のキューサイズ"""
    return _data_queue.qsize()


# ============================================================
# ユーティリティ
# ============================================================

def get_adc_resolution() -> int:
    """現在設定されているADC解像度を取得"""
    return _adc_resolution


def get_adc_max_value() -> int:
    """現在設定されているADC最大値を取得"""
    return _adc_max_value


def adc_to_voltage(adc_value: int, vref: float = 3.3) -> float:
    """
    ADC値を電圧に変換
    
    Args:
        adc_value: ADC値
        vref: 基準電圧 (デフォルト: 3.3V)
    
    Returns:
        電圧値
    """
    return (adc_value / _adc_max_value) * vref
