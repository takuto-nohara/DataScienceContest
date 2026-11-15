function [featureMean, featureStd, unscaled] = scalingConstants(trainRaw)
% 本番の30列（標準化前）を featurePool から計算し、学習データ＋拡張8通りでの
% 平均と標準偏差を返す。拡張は train_final.m と同じ。
names = ergonomicNames();
augmentations = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
windows = [{trainRaw}, cellfun(@(f) f(trainRaw), augmentations, 'UniformOutput', false)];
parts = cellfun(@(w) unscaledFeatures(w, names), windows, 'UniformOutput', false);
unscaled = cat(1, parts{:});
featureMean = mean(unscaled, 1);
featureStd = std(unscaled, 0, 1);
featureStd(featureStd < 1e-12) = 1;
end


function F = unscaledFeatures(raw, names)
[P, poolNames] = featurePool(raw);
F = P(:, arrayfun(@(s) find(poolNames == s), names));
F(:, 14:16) = log10(max(F(:, 14:16), 1e-9));
end


function names = ergonomicNames()
names = ["gravity_dir_" + ["x" "y" "z"], "acc_mag_mean", ...
    "acc_std_" + ["x" "y" "z"], "acc_mag_std", "vertical_std", ...
    "horizontal_mean", "horizontal_std", "acc_xy_motion_std", "acc_max_x", ...
    "gyro_std_" + ["x" "y" "z"], "gyro_mag_mean", ...
    "acc_jerk_rms_" + ["x" "y" "z"], "vertical_jerk_max", "vertical_jerk_min", ...
    "acc_x_band_0p5_3hz", "acc_y_band_0p5_3hz", "gyro_x_band_0p5_3hz", "step_freq_hz", ...
    "acc_corr_xy", "gyro_corr_yz", "gyro_corr_zx", "vertical_skewness"];
end
