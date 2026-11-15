function [x, names, family] = featurePool(raw)
% セミナーで出た特徴量のアイデアをすべて計算し、候補の集まり（プール）を作る。
%
%   raw    : N×128×6 (加速度 x/y/z、角速度 x/y/z)。50 Hz、128点
%   x      : N×P の特徴量
%   names  : 1×P の列名
%   family : 1×P の系統名。同じ系統の列は、まとめて追加・削除して比べる
%
% どの列も、その試行の128点だけから計算する（ほかの試行の値は使わない）。
% 系統の名前の後ろの括弧は、セミナーでその案を出した人の頭文字。

fs = 50;
n = size(raw, 1);
acc = raw(:, :, 1:3);
gyro = raw(:, :, 4:6);
axes3 = ["x", "y", "z"];

x = zeros(n, 0);
names = strings(1, 0);
family = strings(1, 0);

    function add(values, columnNames, familyName)
        x = [x, values];
        names = [names, columnNames];
        family = [family, repmat(string(familyName), 1, numel(columnNames))];
    end

accMean = squeeze3(mean(acc, 2));
gyroMean = squeeze3(mean(gyro, 2));
accStd = squeeze3(std(acc, 0, 2));
gyroStd = squeeze3(std(gyro, 0, 2));

% 主催者の12列（平均と標準偏差）
add(accMean, "acc_mean_" + axes3, "acc_mean");
add(gyroMean, "gyro_mean_" + axes3, "gyro_mean");
add(accStd, "acc_std_" + axes3, "acc_std");
add(gyroStd, "gyro_std_" + axes3, "gyro_std");

% 最大・最小（梶山・松井・新井）
add([squeeze3(max(acc, [], 2)), squeeze3(min(acc, [], 2))], ...
    ["acc_max_" + axes3, "acc_min_" + axes3], "acc_maxmin");
add([squeeze3(max(gyro, [], 2)), squeeze3(min(gyro, [], 2))], ...
    ["gyro_max_" + axes3, "gyro_min_" + axes3], "gyro_maxmin");

% 二乗平均平方根（松井）
add(squeeze3(sqrt(mean(acc .^ 2, 2))), "acc_rms_" + axes3, "acc_rms");

% 3軸の大きさ（野原・新井）
accMag = sqrt(sum(acc .^ 2, 3));
gyroMag = sqrt(sum(gyro .^ 2, 3));
add(magSummary(accMag, fs), "acc_mag_" + ["mean", "std", "max", "jerk_rms"], "acc_mag");
add(magSummary(gyroMag, fs), "gyro_mag_" + ["mean", "std", "max", "change_rms"], "gyro_mag");

% 軸どうしの相関（野原）
add(correlations(acc), "acc_corr_" + ["xy", "yz", "zx"], "acc_corr");
add(correlations(gyro), "gyro_corr_" + ["xy", "yz", "zx"], "gyro_corr");

% 重力の向き（野原・石井）
gravityDir = accMean ./ max(sqrt(sum(accMean .^ 2, 2)), eps);
add(gravityDir, "gravity_dir_" + axes3, "gravity_dir");

% 加速度の変化率＝躍度（野原・石井）と2階差分（新井）
add(squeeze3(sqrt(mean((diff(acc, 1, 2) * fs) .^ 2, 2))), "acc_jerk_rms_" + axes3, "acc_jerk");
add(squeeze3(sqrt(mean((diff(acc, 2, 2) * fs ^ 2) .^ 2, 2))), "acc_diff2_rms_" + axes3, "acc_diff2");

% Z方向の変化量（松井・新井）と XY方向の動き（新井）。端末の座標そのままで計算する
accBody = acc - mean(acc, 2);
add([max(acc(:, :, 3), [], 2) - min(acc(:, :, 3), [], 2), ...
    mean(abs(diff(acc(:, :, 3), 1, 2)), 2)], ["acc_z_range", "acc_z_mean_abs_change"], "z_change");
xyMotion = sqrt(accBody(:, :, 1) .^ 2 + accBody(:, :, 2) .^ 2);
add([mean(xyMotion, 2), std(xyMotion, 0, 2)], ["acc_xy_motion_mean", "acc_xy_motion_std"], "xy_motion");

% 0.5〜3 Hz のパワーの割合（既存の spectral32）
add(bandFraction(raw, fs), ["acc_", "acc_", "acc_", "gyro_", "gyro_", "gyro_"] ...
    + ["x", "y", "z", "x", "y", "z"] + "_band_0p5_3hz", "band_frac");

% 重力方向（上下）と水平方向に分けた動き。端末の向きに左右されにくい
vertical = sum(accBody .* reshape(gravityDir, n, 1, 3), 3);
horizontal = sqrt(max(sum(accBody .^ 2, 3) - vertical .^ 2, 0));
verticalJerk = diff(vertical, 1, 2) * fs;
add([std(vertical, 0, 2), std(horizontal, 0, 2), mean(horizontal, 2), ...
    max(verticalJerk, [], 2), min(verticalJerk, [], 2), skewness(vertical, 1, 2)], ...
    ["vertical_std", "horizontal_std", "horizontal_mean", ...
    "vertical_jerk_max", "vertical_jerk_min", "vertical_skewness"], "vert_horiz");

% 歩調：加速度の大きさの卓越周波数と、スペクトルの散らばり
add(stepFrequency(accMag, fs), ["step_freq_hz", "acc_mag_spectral_entropy"], "step_freq");

% 角速度の絶対値の平均（回転の量。向きの符号に左右されない）
add(squeeze3(mean(abs(gyro), 2)), "gyro_absmean_" + axes3, "gyro_absmean");

assert(all(isfinite(x(:))));
end


function y = squeeze3(v)
y = reshape(v, size(v, 1), size(v, 3));
end


function s = magSummary(signal, fs)
s = [mean(signal, 2), std(signal, 0, 2), max(signal, [], 2), ...
    sqrt(mean((diff(signal, 1, 2) * fs) .^ 2, 2))];
end


function r = correlations(threeAxes)
pairs = [1 2; 2 3; 3 1];
r = zeros(size(threeAxes, 1), 3);
for k = 1:3
    a = threeAxes(:, :, pairs(k, 1)) - mean(threeAxes(:, :, pairs(k, 1)), 2);
    b = threeAxes(:, :, pairs(k, 2)) - mean(threeAxes(:, :, pairs(k, 2)), 2);
    d = sqrt(sum(a .^ 2, 2) .* sum(b .^ 2, 2));
    c = sum(a .* b, 2) ./ max(d, eps);
    c(d < 1e-12) = 0;
    r(:, k) = c;
end
end


function f = bandFraction(raw, fs)
n = size(raw, 1);
hz = (1:64) * fs / 128;
p = abs(fft(raw - mean(raw, 2), [], 2)) .^ 2;
p = p(:, 2:65, :);
inBand = hz >= 0.5 & hz <= 3;
f = reshape(sum(p(:, inBand, :), 2) ./ max(sum(p, 2), eps), n, 6);
end


function f = stepFrequency(magnitude, fs)
hz = (1:64) * fs / 128;
p = abs(fft(magnitude - mean(magnitude, 2), [], 2)) .^ 2;
p = p(:, 2:65);
[~, peak] = max(p, [], 2);
freq = hz(peak)';
freq(std(magnitude, 0, 2) < 1e-9) = 0;
prob = p ./ max(sum(p, 2), eps);
entropy = -sum(prob .* log(max(prob, eps)), 2) / log(64);
f = [freq, entropy];
end
