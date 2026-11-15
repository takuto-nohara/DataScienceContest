function stage1_compare(root)
% 第1段：特徴量の候補ごとの分離のよさと、特徴量セット×手法の比較。
% 評価は学習用2941窓の記録単位5分割交差検証。テスト201窓は選択に使わず参考値として記録する。
if nargin < 1
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
if ~isfolder(out), mkdir(out); end

data = har.loadData(root);
[Xtr, names, family] = featurePool(data.trainRaw);
Xte = featurePool(data.testRaw);
Ytr = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, data.trainRaw, data.testRaw));
gtr = groups(1:numel(Ytr));
save(fullfile(out, 'pool.mat'), 'Xtr', 'Xte', 'Ytr', 'Yte', 'names', 'family', 'gtr');
fprintf('pool: %d features, %d families, %d train recordings\n', ...
    numel(names), numel(unique(family)), numel(unique(gtr)));

%% 1. 行動ごとの平均と、紛らわしい組ごとの分離度（フィッシャー比）
pairs = {'Sitting', 'Standing'; 'WalkingUpstairs', 'WalkingDownstairs'; ...
    'Walking', 'WalkingUpstairs'; 'Walking', 'WalkingDownstairs'; ...
    'Laying', 'Sitting'};
static = ismember(Ytr, categorical(["Sitting", "Standing", "Laying"]));
T = table(names', family', 'VariableNames', {'feature', 'family'});
for c = 1:numel(data.classes)
    T.("mean_" + data.classes(c)) = mean(Xtr(Ytr == data.classes(c), :), 1)';
end
T.fisher_static_vs_dynamic = fisher(Xtr(static, :), Xtr(~static, :));
for k = 1:size(pairs, 1)
    a = Xtr(Ytr == pairs{k, 1}, :);
    b = Xtr(Ytr == pairs{k, 2}, :);
    T.("fisher_" + pairs{k, 1} + "_vs_" + pairs{k, 2}) = fisher(a, b);
end
writetable(T, fullfile(out, 'feature_separability.csv'));

%% 2. 特徴量セット×手法
sets = struct();
sets.stats12 = familyCols(family, ["acc_mean", "gyro_mean", "acc_std", "gyro_std"]);
sets.hirano9 = familyCols(family, ["acc_mean", "acc_std", "gyro_std"]);
sets.matsui30 = familyCols(family, ["acc_mean", "gyro_mean", "acc_std", "gyro_std", ...
    "acc_maxmin", "gyro_maxmin", "acc_rms"]);
sets.arai32 = familyCols(family, ["acc_mean", "gyro_mean", "acc_std", "gyro_std", ...
    "acc_maxmin", "acc_mag", "gyro_mag", "z_change", "xy_motion"]);
sets.nohara_like = familyCols(family, ["acc_mean", "gyro_mean", "acc_std", "gyro_std", ...
    "acc_mag", "gyro_mag", "acc_corr", "gyro_corr", "gravity_dir", "acc_jerk"]);
sets.spectral32 = familyCols(family, ["acc_mean", "gyro_mean", "acc_std", "gyro_std", ...
    "acc_mag", "gyro_mag", "acc_corr", "gyro_corr", "band_frac"]);
sets.pool_all = 1:numel(names);

methods = ["tree", "lda", "knn", "subspace_knn", "linear_svm", "gauss_svm", ...
    "bagged_tree", "nnet"];
setNames = string(fieldnames(sets))';
rows = {};
for s = setNames
    cols = sets.(s);
    for m = methods
        tic;
        [cvAcc, pred] = groupedCV(Xtr(:, cols), Ytr, gtr, char(m), 1);
        rng(7, 'twister');
        f = fitModel(char(m), Xtr(:, cols), Ytr);
        testAcc = mean(f(Xte(:, cols)) == Yte);
        sec = toc;
        sit = confusionPair(Ytr, pred, 'Sitting', 'Standing');
        stair = confusionPair(Ytr, pred, 'WalkingUpstairs', 'WalkingDownstairs');
        fprintf('%-12s %-13s p=%2d  CV %.4f  test %.4f  sit/stand err %3d  up/down err %3d  (%.0fs)\n', ...
            s, m, numel(cols), cvAcc, testAcc, sit, stair, sec);
        rows(end + 1, :) = {s, m, numel(cols), cvAcc, testAcc, sit, stair, sec}; %#ok<AGROW>
    end
end
R = cell2table(rows, 'VariableNames', {'featureSet', 'method', 'nFeatures', ...
    'groupedCvAccuracy', 'test201Accuracy', 'cvSitStandErrors', 'cvUpDownErrors', 'seconds'});
writetable(R, fullfile(out, 'stage1_set_by_method.csv'));
fprintf('STAGE1_DONE\n');
end


function f = fisher(a, b)
f = ((mean(a, 1) - mean(b, 1)) .^ 2 ./ max(var(a, 0, 1) + var(b, 0, 1), eps))';
end


function cols = familyCols(family, wanted)
cols = find(ismember(family, wanted));
end


function e = confusionPair(Y, P, a, b)
e = sum(Y == a & P == b) + sum(Y == b & P == a);
end
