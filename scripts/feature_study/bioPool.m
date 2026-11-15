function [x, names, family] = bioPool(raw)
% 査読付き論文の生体計測の考え方から作った、座位と立位を見分ける候補特徴量。
% どの列も、その試行の128点（50 Hz、2.56秒）だけから計算する。
%
%   frame    : 加速度の平均から重力方向を求め、センサの軸を「鉛直」と「水平2軸」に回す。
%              腰の装着角度に左右されない形で揺れを測るため
%   ape      : 鉛直からの体幹（骨盤）の傾き角。腰のセンサで姿勢を角度で判定する方法
%              （Vähä-Ypyä ら 2018, Scand J Med Sci Sports）。座位では骨盤が後ろに倒れる
%              （Endo ら 2012, J Orthop Sci）
%   sway     : 水平面の揺れの大きさ・平均周波数・95%周波数・楕円の面積
%              （体幹加速度による静止立位の動揺の指標。Moe-Nilssen & Helbostad 2002,
%              Gait Posture; Mancini ら 2011, Parkinsonism Relat Disord）
%   rotation : 重力の向きに合わせたジャイロ。鉛直軸まわりのねじれと、水平軸まわりの傾きの速さ
%   pivot    : 倒立振子の支点までの距離。立位は足首を支点に全身が揺れる
%              （Winter ら 1998, J Neurophysiol）ので、腰は支点から約1 m 上にあり、
%              傾きの角加速度に比例した水平の加速度が出る。座位は骨盤（座面）が支点
%              （Cholewicki ら 2000, J Biomech）で、腰は支点のすぐ上にある
%   cardiac  : 心拍の周期。起立すると心拍は上がる（起立時の心拍反応）。
%              体の微小な振動の包絡線の自己相関から周期の強さと拍数を求める
%              （心弾動図。Inan ら 2015, IEEE J Biomed Health Inform）
fs = 50;
n = size(raw, 1);
T = size(raw, 2);
G = 9.80665;
acc = raw(:, :, 1:3);
gyro = raw(:, :, 4:6);
lg = @(v) log10(max(v, 1e-9));
x = zeros(n, 0); names = strings(1, 0); family = strings(1, 0);
    function add(values, columnNames, familyName)
        x = [x, values];
        names = [names, columnNames];
        family = [family, repmat(string(familyName), 1, numel(columnNames))];
    end

%% 重力方向に合わせた座標（鉛直 ev、水平 e1 は y 軸を水平面に写したもの、e2 = ev × e1）
m = reshape(mean(acc, 2), n, 3);
ev = m ./ max(vecnorm(m, 2, 2), eps);
yAxis = repmat([0 1 0], n, 1);
e1 = yAxis - sum(yAxis .* ev, 2) .* ev;
e1 = e1 ./ max(vecnorm(e1, 2, 2), eps);
e2 = cross(ev, e1, 2);
proj = @(s, e) sum(s .* reshape(e, n, 1, 3), 3);
accC = acc - mean(acc, 2);
h1 = proj(accC, e1) * G;           % 水平の揺れ（m/s^2）
h2 = proj(accC, e2) * G;
wv = proj(gyro, ev);               % 鉛直軸まわりの回転（ねじれ）
w1 = proj(gyro, e1);               % 水平軸まわりの回転（傾き）
w2 = proj(gyro, e2);

%% ape: 鉛直（センサ x 軸）からの傾き角と、前後・左右の成分
add([acosd(min(1, abs(ev(:, 1)))), atan2d(ev(:, 2), ev(:, 1)), atan2d(ev(:, 3), ev(:, 1))], ...
    ["ape_incl_deg", "ape_sagittal_deg", "ape_frontal_deg"], "ape");

%% sway: 水平面の揺れ
hz = (0:T / 2) * fs / T;
band = hz >= 0.2 & hz <= 5;
    function [rmsv, fmean, f95] = spectral(s)
        p = abs(fft(s - mean(s, 2), [], 2)) .^ 2;
        p = p(:, 1:T / 2 + 1);
        pb = p(:, band);
        f = hz(band);
        tot = max(sum(pb, 2), eps);
        rmsv = sqrt(mean(s .^ 2, 2));
        fmean = (pb * f') ./ tot;
        c = cumsum(pb, 2) ./ tot;
        [~, k] = max(c >= 0.95, [], 2);
        f95 = f(k)';
    end
[r1, fm1, f951] = spectral(h1);
[r2, fm2, f952] = spectral(h2);
hr = sqrt(h1 .^ 2 + h2 .^ 2);
c11 = mean(h1 .^ 2, 2); c22 = mean(h2 .^ 2, 2); c12 = mean(h1 .* h2, 2);
ellipse = pi * 5.991 * sqrt(max(c11 .* c22 - c12 .^ 2, 0));   % 95% 信頼楕円の面積
add([lg(r1), lg(r2), lg(mean(hr, 2)), lg(ellipse)], ...
    ["log_sway_rms_ap", "log_sway_rms_ml", "log_sway_mean_dist", "log_sway_ellipse"], "sway_size");
add([fm1, fm2, f951, f952], ["sway_fmean_ap", "sway_fmean_ml", "sway_f95_ap", "sway_f95_ml"], "sway_freq");
add(lg(sum(abs(diff(h1, 1, 2)) + abs(diff(h2, 1, 2)), 2)), "log_sway_path", "sway_size");

%% rotation: 重力に合わせたジャイロ
[rv, fmv] = spectral(wv);
[rw1, fmw1] = spectral(w1);
[rw2, fmw2] = spectral(w2);
add([lg(rv), lg(rw1), lg(rw2), lg(sqrt(rw1 .^ 2 + rw2 .^ 2))], ...
    ["log_yaw_rate_rms", "log_tilt_rate_rms_1", "log_tilt_rate_rms_2", "log_tilt_rate_rms"], "rotation_size");
add([fmv, fmw1, fmw2], ["yaw_fmean", "tilt_fmean_1", "tilt_fmean_2"], "rotation_freq");
add(lg(rv) - lg(sqrt(rw1 .^ 2 + rw2 .^ 2)), "log_yaw_to_tilt", "rotation_ratio");

%% pivot: 倒立振子の支点までの距離
% 水平の加速度 = 支点までの距離 × 角加速度 ＋ 重力 × 傾き角 ＋ ゆっくりした変化
% を窓ごとに最小二乗で当てはめ、角加速度の係数（m）を支点までの距離とする。
sm = @(s) movmean(s, 5, 2);
tt = ((1:T) - (T + 1) / 2) / fs;
hPivot = zeros(n, 2); r2gain = zeros(n, 2);
pairs = {sm(h2), sm(w1), -1; sm(h1), sm(w2), 1};   % e1 まわりの回転は e2 方向、e2 まわりは e1 方向
for q = 1:2
    a = pairs{q, 1}; w = pairs{q, 2}; sgn = pairs{q, 3};
    theta = cumsum(w - mean(w, 2), 2) / fs;
    alpha = gradient(w) * fs;
    for i = 1:n
        A0 = [ones(T, 1), tt', theta(i, :)'];
        A = [A0, alpha(i, :)'];
        y = a(i, :)';
        b = pinv(A) * y;
        b0 = pinv(A0) * y;
        res = sum((y - A * b) .^ 2);
        res0 = sum((y - A0 * b0) .^ 2);
        hPivot(i, q) = sgn * b(4);
        r2gain(i, q) = 1 - res / max(res0, eps);
    end
end
add([hPivot, mean(hPivot, 2), sign(mean(hPivot, 2)) .* lg(abs(mean(hPivot, 2)))], ...
    ["pivot_height_1", "pivot_height_2", "pivot_height_mean", "pivot_height_slog"], "pivot");
add([r2gain, mean(r2gain, 2)], ["pivot_r2_1", "pivot_r2_2", "pivot_r2_mean"], "pivot_fit");

%% cardiac: 微小な振動の包絡線の周期（40〜150拍/分 = 遅れ 0.4〜1.5 秒）
accMag = sqrt(sum(acc .^ 2, 3));
hp = accMag - movmean(accMag, 7, 2);               % 約7 Hz 以上を残す
env = movmean(abs(hp), 5, 2);
env = env - mean(env, 2);
lags = round(0.4 * fs):round(1.5 * fs);
ac = zeros(n, numel(lags));
den = max(sum(env .^ 2, 2), eps);
for k = 1:numel(lags)
    L = lags(k);
    ac(:, k) = sum(env(:, 1:end - L) .* env(:, 1 + L:end), 2) ./ den * T / (T - L);
end
[peak, kp] = max(ac, [], 2);
add([peak, 60 * fs ./ lags(kp)'], ["cardiac_periodicity", "cardiac_rate_bpm"], "cardiac");

x(~isfinite(x)) = 0;
end
