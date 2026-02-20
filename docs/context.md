# 洗練版 EOG 特徴量セット（提案）

## 1. 位置（方向・中央からの距離・揺らぎ）

- feat_mean_h = mean(ch1)
- feat_mean_v = mean(ch3)
- feat_median_h = median(ch1)
- feat_median_v = median(ch3)

- feat_std_h = std(ch1)
- feat_std_v = std(ch3)

- feat_gaze_radius = sqrt(feat_mean_h**2 + feat_mean_v**2)
- feat_gaze_angle = atan2(feat_mean_v, feat_mean_h)

## 2. 位置トレンド・変化（移動の向きと大きさ）

- t = 0, 1, ..., T-1 （ウィンドウ内サンプルインデックス）

- feat_slope_h = slope of linear regression ch1 ~ t
- feat_slope_v = slope of linear regression ch3 ~ t

- feat_delta_h = ch1[T-1] - ch1[0]
- feat_delta_v = ch3[T-1] - ch3[0]

## 3. サッカード総量・方向性

### 3.1 総量

- feat_sac_energy_h = mean(abs(ch2))
- feat_sac_energy_v = mean(abs(ch4))
- feat_sac_energy_total = feat_sac_energy_h + feat_sac_energy_v + ε

### 3.2 方向比

- feat_sac_energy_ratio_h = feat_sac_energy_h / feat_sac_energy_total
- feat_sac_energy_ratio_v = feat_sac_energy_v / feat_sac_energy_total

### 3.3 イベント数

ピーク検出により peaks_h, peaks_v を得る（ch2, ch4 上）。

- feat_n_sac_h = len(peaks_h)
- feat_n_sac_v = len(peaks_v)

## 4. サッカード時間局在（前半/後半）

ウィンドウを前半 [0, ..., floor(T/2)-1] と後半 [floor(T/2), ..., T-1] に分割。

- feat_energy_sac_h_first = mean(abs(ch2[0 : floor(T/2)]))
- feat_energy_sac_h_last = mean(abs(ch2[floor(T/2) : T]))

- feat_energy_sac_v_first = mean(abs(ch4[0 : floor(T/2)]))
- feat_energy_sac_v_last = mean(abs(ch4[floor(T/2) : T]))

## 5. 「移動中」スコア（位置変化 × サッカード）

- feat_motion_score_h = abs(feat_slope_h) \* feat_sac_energy_h
- feat_motion_score_v = abs(feat_slope_v) \* feat_sac_energy_v

- feat_motion_score = sqrt(feat_slope_h**2 + feat_slope_v**2) \* feat_sac_energy_total

## 6. （必要になった場合のみ）時間インデックス

ウィンドウ ID を w、全ウィンドウ数を W_max とする。

- feat_t_step = w
- feat_t_norm = w / (W_max - 1)
