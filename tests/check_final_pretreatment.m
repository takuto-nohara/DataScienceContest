% 本番候補の前処理関数 submission/ergonomic32_svm を検査する。
%   1. 出力が 試行数×32 の実数行列で、すべて有限
%   2. 32列が、実験用の productionFeatures（featurePool の30列＋ジャイロxの差分2列）と一致する
%      （列 14-16 の回転の揺れは gyro_std の常用対数。どの列も、学習データ＋拡張8通りで
%       計算し直した平均と標準偏差で標準化してから比べる。定数が正しいことも同時に確かめる）
%   3. 1試行ずつ渡しても、まとめて渡しても同じ値になる（試行ごとに独立）
%   4. 値が変化しない試行（全点同じ値）でも有限の値を返す
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'), fullfile(root, 'scripts', 'feature_study'));
addpath(fullfile(root, 'submission', 'ergonomic32_svm'));

data = har.loadData(root);
raw = cat(1, data.trainRaw, data.testRaw);
args = arrayfun(@(c) raw(:, :, c), 1:6, 'UniformOutput', false);
X = Pretreatment_MATLABcontest_Advanced(args{:});
assert(isequal(size(X), [size(raw, 1), 32]) && isreal(X) && all(isfinite(X(:))), 'shape');

finalNames = ["gravity_dir_" + ["x" "y" "z"], "acc_mag_mean", ...
    "acc_std_" + ["x" "y" "z"], "acc_mag_std", "vertical_std", ...
    "horizontal_mean", "horizontal_std", "acc_xy_motion_std", "acc_max_x", ...
    "gyro_std_" + ["x" "y" "z"], "gyro_mag_mean", ...
    "acc_jerk_rms_" + ["x" "y" "z"], "vertical_jerk_max", "vertical_jerk_min", ...
    "acc_x_band_0p5_3hz", "acc_y_band_0p5_3hz", "gyro_x_band_0p5_3hz", "step_freq_hz", ...
    "acc_corr_xy", "gyro_corr_yz", "gyro_corr_zx", "vertical_skewness"];
[P, names] = featurePool(raw);
cols = arrayfun(@(s) find(names == s), finalNames);
expected = productionFeatures(raw);
assert(isequal(expected(:, 1:30), [P(:, cols(1:13)), log10(max(P(:, cols(14:16)), 1e-9)), P(:, cols(17:30))]), 'pool');
[featureMean, featureStd] = scalingConstants(data.trainRaw);
expected = (expected - featureMean) ./ featureStd;
difference = max(abs(X - expected), [], 1) ./ max(1, max(abs(expected), [], 1));
columnNames = [finalNames, "gyro_x_diff_std", "gyro_x_diff2_std"];
assert(all(difference < 1e-9), 'mismatch in %s', strjoin(columnNames(difference >= 1e-9), ', '));

rows = [1, 7, 500, 2941, 3000, size(raw, 1)];
for r = rows
    one = arrayfun(@(c) raw(r, :, c), 1:6, 'UniformOutput', false);
    assert(max(abs(Pretreatment_MATLABcontest_Advanced(one{:}) - X(r, :))) < 1e-12, 'batch dependence');
end

constant = arrayfun(@(c) ones(2, 128) * c * 0.1, 1:6, 'UniformOutput', false);
Z = Pretreatment_MATLABcontest_Advanced(constant{:});
assert(all(isfinite(Z(:))), 'constant input');

fprintf('FINAL_PRETREATMENT_CHECK_PASSED (%d trials, max relative difference %.2e)\n', ...
    size(raw, 1), max(difference));
