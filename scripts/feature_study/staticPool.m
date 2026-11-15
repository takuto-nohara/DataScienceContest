function [x, names, family] = staticPool(raw)
% 静止姿勢（座位・立位・臥位）を見分けるための、微小な動きの候補特徴量。
% 静止中の揺れや躍度は動作中の100分の1程度しかないため、対数も合わせて計算する。
%
%   躍度（加加速度）: 加速度の時間微分。姿勢を保つための筋の小さな修正動作に現れる
%   スナップ        : 加速度の2階微分
%   帯域パワー       : 姿勢動揺（0.3〜2 Hz）、中間（2〜6 Hz）、生理的振戦（6〜12 Hz）、
%                     高域（12〜25 Hz）
%   傾き角の揺れ     : 重力方向の角度が窓の中でどれだけ揺れるか
% どの列も、その試行の128点だけから計算する。
fs = 50;
n = size(raw, 1);
acc = raw(:, :, 1:3);
gyro = raw(:, :, 4:6);
axes3 = ["x", "y", "z"];
lg = @(v) log10(max(v, 1e-9));
sq = @(v) reshape(v, n, []);

x = zeros(n, 0); names = strings(1, 0); family = strings(1, 0);
    function add(values, columnNames, familyName)
        x = [x, values];
        names = [names, columnNames];
        family = [family, repmat(string(familyName), 1, numel(columnNames))];
    end

accMean = sq(mean(acc, 2));
g = accMean ./ max(sqrt(sum(accMean .^ 2, 2)), eps);
add(g, "gravity_dir_" + axes3, "posture");
add(sqrt(sum(accMean .^ 2, 2)), "acc_mean_norm", "posture_norm");
add([atan2d(g(:, 2), g(:, 1)), atan2d(g(:, 3), g(:, 1))], ["tilt_y_deg", "tilt_z_deg"], "posture_angle");

% 躍度とスナップ（生の値と対数）
jerk = diff(acc, 1, 2) * fs;
snap = diff(acc, 2, 2) * fs ^ 2;
gyroJerk = diff(gyro, 1, 2) * fs;
jr = sq(sqrt(mean(jerk .^ 2, 2)));
sr = sq(sqrt(mean(snap .^ 2, 2)));
gjr = sq(sqrt(mean(gyroJerk .^ 2, 2)));
jmag = sqrt(sum(jerk .^ 2, 3));
add(jr, "jerk_rms_" + axes3, "jerk_raw");
add(lg(jr), "log_jerk_rms_" + axes3, "jerk_log");
add([lg(mean(jmag, 2)), lg(max(jmag, [], 2)), lg(prctile(jmag, 90, 2))], ...
    ["log_jerk_mag_mean", "log_jerk_mag_max", "log_jerk_mag_p90"], "jerk_mag_log");
add(lg(sr), "log_snap_rms_" + axes3, "snap_log");
add(lg(gjr), "log_gyro_jerk_rms_" + axes3, "gyro_jerk_log");
% 躍度の尖り（短い修正が間欠的に入るか、連続的な雑音か）
jc = jmag - mean(jmag, 2);
add(mean(jc .^ 4, 2) ./ max(mean(jc .^ 2, 2) .^ 2, eps), "jerk_mag_kurtosis", "jerk_shape");

% 揺れの大きさ（対数）
add(lg(sq(std(acc, 0, 2))), "log_acc_std_" + axes3, "std_log");
add(lg(sq(std(gyro, 0, 2))), "log_gyro_std_" + axes3, "std_log");

% 帯域パワー（加速度の大きさとジャイロの大きさ、対数）
hz = (1:64) * fs / 128;
bands = [0.3 2; 2 6; 6 12; 12 25];
accMag = sqrt(sum(acc .^ 2, 3));
gyroMag = sqrt(sum(gyro .^ 2, 3));
for sig = {accMag, "accmag"; gyroMag, "gyromag"}'
    s = sig{1};
    p = abs(fft(s - mean(s, 2), [], 2)) .^ 2;
    p = p(:, 2:65);
    for b = 1:size(bands, 1)
        in = hz >= bands(b, 1) & hz < bands(b, 2);
        add(lg(sum(p(:, in), 2) / 128 ^ 2), sprintf("log_%s_power_%g_%ghz", sig{2}, bands(b, 1), bands(b, 2)), "band_log");
    end
end
% ジャイロ各軸の姿勢動揺帯域
for a = 1:3
    p = abs(fft(gyro(:, :, a) - mean(gyro(:, :, a), 2), [], 2)) .^ 2;
    p = p(:, 2:65);
    in = hz >= 0.3 & hz < 2;
    add(lg(sum(p(:, in), 2) / 128 ^ 2), "log_gyro_" + axes3(a) + "_power_0.3_2hz", "gyro_sway_log");
end

% 傾き角の窓内の揺れ（重力方向がどれだけふらつくか）
lowpass = movmean(acc, 25, 2);
inst = lowpass ./ max(sqrt(sum(lowpass .^ 2, 3)), eps);
ang = acosd(min(1, sum(inst .* reshape(g, n, 1, 3), 3)));
add([lg(std(ang, 0, 2)), lg(max(ang, [], 2))], ["log_tilt_sway_std", "log_tilt_sway_max"], "tilt_sway");

% ジャイロと加速度の揺れの比（回転で揺れるか、平行移動で揺れるか）
add(lg(sqrt(sum(sq(var(gyro, 0, 2)), 2))) - lg(sqrt(sum(sq(var(acc, 0, 2)), 2))), ...
    "log_gyro_to_acc_sway_ratio", "sway_ratio");

x(~isfinite(x)) = 0;
end
