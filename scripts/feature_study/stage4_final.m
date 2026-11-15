function stage4_final(root, mode)
% 第4段：本番向けの最終候補を選ぶ。
%   目的関数 = 「ずれなし」と「現実的なずれ5種」（装着角度10°、動きの強さ0.7/1.3倍、
%   歩調0.85/1.15倍）の正解率の平均。ずれは評価側の窓だけに加え、学習側の拡張とは
%   別の乱数・別の倍率を使う。
%   mode = 'sets' : 特徴量セットの比較（手法×拡張）
%          'grid' : 最良セットでのハイパーパラメータの平坦さの確認
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2, mode = 'sets'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
cacheFile = fullfile(out, 'stage4_cache.mat');

if isfile(cacheFile)
    load(cacheFile, 'C');
else
    C = buildCache(root);
    save(cacheFile, 'C', '-v7.3');
end
if ~isfield(C, 'XaugNoise')
    % 雑音を加えた拡張（静止時の揺れと同程度の 0.005）を2通り
    data = har.loadData(root);
    C.XaugNoise = {featurePool(perturb(data.trainRaw, 'noise', 0.005, 21)), ...
        featurePool(perturb(data.trainRaw, 'noise', 0.005, 22))};
    save(cacheFile, 'C', '-v7.3');
end
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'')', strrep(here, '\', '/')));

sets = candidateSets(C.names, out);
switch mode
    case 'sets'
        setNames = string(fieldnames(sets))';
        jobs = {};
        for s = setNames
            for m = ["gauss_svm", "knn", "bagged_tree"]
                for a = ["none", "full"]
                    jobs(end + 1, :) = {s, m, a, struct()}; %#ok<AGROW>
                end
            end
        end
    case 'grid'
        jobs = {};
        s = "ergonomic32";
        p = numel(sets.(s));
        for a = ["full", "full_noise"]
            for ks = [0.5, 0.7, 1, 1.4, 2]
                for box = [1, 3, 10, 30, 100]
                    jobs(end + 1, :) = {s, "gauss_svm", a, ...
                        struct('KernelScale', ks * sqrt(p), 'BoxConstraint', box)}; %#ok<AGROW>
                end
            end
        end
end

res = cell(size(jobs, 1), 1);
parfor j = 1:size(jobs, 1)
    res{j} = evaluate(C, sets.(jobs{j, 1}), char(jobs{j, 2}), jobs{j, 3}, jobs{j, 4});
end

rows = {};
for j = 1:size(jobs, 1)
    r = res{j};
    o = jobs{j, 4};
    ks = NaN; box = NaN;
    if isfield(o, 'KernelScale'), ks = o.KernelScale / sqrt(numel(sets.(jobs{j, 1}))); end
    if isfield(o, 'BoxConstraint'), box = o.BoxConstraint; end
    rows(end + 1, :) = [{jobs{j, 1}, jobs{j, 2}, jobs{j, 3}, numel(sets.(jobs{j, 1})), ks, box, ...
        r.objective, r.clean, r.cleanSd, r.test201}, num2cell(r.shift), ...
        {r.sitStand, r.upDown}]; %#ok<AGROW>
    fprintf('%-14s %-11s %-4s ks %4.2f C %5.1f  J %.4f clean %.4f (sd %.4f) t201 %.4f | %s | sit/stand %d up/down %d\n', ...
        jobs{j, 1}, jobs{j, 2}, jobs{j, 3}, ks, box, r.objective, r.clean, r.cleanSd, r.test201, ...
        sprintf('%.3f ', r.shift), r.sitStand, r.upDown);
end
T = cell2table(rows, 'VariableNames', [{'featureSet', 'method', 'augment', 'nFeatures', ...
    'kernelScaleOverSqrtP', 'boxConstraint', 'objective', 'cleanCv', 'cleanCvSd', 'test201'}, ...
    cellstr("shift_" + C.shiftNames), {'cvSitStandErrors', 'cvUpDownErrors'}]);
writetable(T, fullfile(out, sprintf('stage4_%s.csv', mode)));
fprintf('STAGE4_DONE\n');
end


function C = buildCache(root)
% 学習側の拡張と評価側のずれを、すべて特徴量プールに変換しておく。
data = har.loadData(root);
raw = data.trainRaw;
C.Y = categorical(data.trainLabels, data.classes);
C.Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
C.g = groups(1:numel(C.Y));
[C.X, C.names] = featurePool(raw);
C.Xte = featurePool(data.testRaw);

% 学習側の拡張（6通り）。行の並びは元の窓と同じ
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1)};
C.Xaug = cellfun(@featurePool, aug, 'UniformOutput', false);

% 評価側のずれ（学習側と違う乱数・倍率）。最初の5つが目的関数に入る
C.shiftNames = ["rotate10", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", ...
    "rotate20", "noise0p01"];
sh = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), ...
    perturb(raw, 'rotate', 20, 902), perturb(raw, 'noise', 0.01, 903)};
C.Xshift = cellfun(@featurePool, sh, 'UniformOutput', false);

% 全学習データで学習するとき用（テスト201窓の参考値）
end


function r = evaluate(C, cols, method, augment, opts)
ids = unique(C.g);
[~, pos] = ismember(C.g, ids);
repeats = 3;
clean = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(C.Xshift));
predFirst = C.Y;
for rep = 1:repeats
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = C.Y;
    for k = 1:5
        tr = fold ~= k;
        [Xtr, Ytr] = trainingRows(C, tr, cols, augment);
        rng(rep * 100 + k, 'twister');
        f = fitModel(method, Xtr, Ytr, opts);
        h = ~tr;
        pred(h) = f(C.X(h, cols));
        if rep == 1
            for q = 1:numel(C.Xshift)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(C.Xshift{q}(h, cols)) == C.Y(h));
            end
        end
    end
    clean(rep) = mean(pred == C.Y);
    if rep == 1, predFirst = pred; end
end
r.clean = mean(clean);
r.cleanSd = std(clean);
r.shift = shiftCorrect / numel(C.Y);
r.objective = mean([r.clean, r.shift(1:5)]);
r.sitStand = sum(C.Y == 'Sitting' & predFirst == 'Standing') + sum(C.Y == 'Standing' & predFirst == 'Sitting');
r.upDown = sum(C.Y == 'WalkingUpstairs' & predFirst == 'WalkingDownstairs') ...
    + sum(C.Y == 'WalkingDownstairs' & predFirst == 'WalkingUpstairs');
[Xall, Yall] = trainingRows(C, true(size(C.Y)), cols, augment);
rng(7, 'twister');
f = fitModel(method, Xall, Yall, opts);
r.test201 = mean(f(C.Xte(:, cols)) == C.Yte);
end


function [X, Y] = trainingRows(C, tr, cols, augment)
X = C.X(tr, cols);
Y = C.Y(tr);
if startsWith(augment, "full")
    extra = C.Xaug;
    if augment == "full_noise", extra = [extra, C.XaugNoise]; end
    for a = 1:numel(extra)
        X = [X; extra{a}(tr, cols)]; %#ok<AGROW>
        Y = [Y; C.Y(tr)]; %#ok<AGROW>
    end
end
end


function sets = candidateSets(names, out)
pick = @(list) arrayfun(@(s) find(names == s), string(list));
fwd = @(m, n) readFwd(out, m, n);

sets.spectral32 = pick(["acc_mean_" + ["x" "y" "z"], "gyro_mean_" + ["x" "y" "z"], ...
    "acc_std_" + ["x" "y" "z"], "gyro_std_" + ["x" "y" "z"], ...
    "acc_mag_" + ["mean" "std" "max" "jerk_rms"], "gyro_mag_" + ["mean" "std" "max" "change_rms"], ...
    "acc_corr_" + ["xy" "yz" "zx"], "gyro_corr_" + ["xy" "yz" "zx"], ...
    ["acc_x" "acc_y" "acc_z" "gyro_x" "gyro_y" "gyro_z"] + "_band_0p5_3hz"]);
sets.svm_fwd20 = pick(fwd('gauss_svm', 20));
sets.knn_fwd14 = pick(fwd('knn', 14));

% 人体工学の役割ごとに選んだ32列（重複する系統は代表だけ残す）
sets.ergonomic32 = pick([ ...
    "gravity_dir_" + ["x" "y" "z"], "acc_mag_mean", ...                       % 姿勢 4
    "acc_std_" + ["x" "y" "z"], "acc_mag_std", "vertical_std", ...            % 強さ（加速度）
    "horizontal_mean", "horizontal_std", "acc_xy_motion_std", "acc_max_x", ...
    "gyro_std_" + ["x" "y" "z"], "gyro_mag_mean", ...                         % 強さ（回転・揺れ）
    "acc_jerk_rms_" + ["x" "y" "z"], "vertical_jerk_max", "vertical_jerk_min", ... % 衝撃
    "acc_x_band_0p5_3hz", "acc_y_band_0p5_3hz", "gyro_x_band_0p5_3hz", "step_freq_hz", ... % リズム
    "acc_corr_xy", "gyro_corr_yz", "gyro_corr_zx", "vertical_skewness"]);       % 協調
% 姿勢と揺れだけを強めた小さいセット（最小構成の確認用）
sets.core12 = pick(["gravity_dir_" + ["x" "y" "z"], "acc_max_x", "acc_jerk_rms_y", ...
    "acc_xy_motion_std", "gyro_corr_yz", "horizontal_mean", "gyro_mag_change_rms", ...
    "gyro_x_band_0p5_3hz", "gyro_std_x", "vertical_jerk_max"]);
end


function f = readFwd(out, m, n)
T = readtable(fullfile(out, sprintf('stage2_forward_%s.csv', m)), 'TextType', 'string');
f = T.addedFeature(1:n)';
end
