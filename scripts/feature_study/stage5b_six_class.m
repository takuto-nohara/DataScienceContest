function stage5b_six_class(root, mode)
% 第5段の2：座位/立位のループで効いた「静止中の揺れ」の列を、6クラスの本番候補
% （ergonomic32 の30列＋空き2列）に入れて、ずれへの強さまで含めて比べる。
%   比べ方は第4段と同じ：目的関数 J = ずれなしと現実的なずれ5種の正解率の平均。
%   学習側は拡張8通り（本番と同じ）、手法はガウスSVM（カーネルスケール √p、ボックス制約 10）。
% 加速度の躍度（対数）も、6クラスで効くかを確かめるために候補に入れる。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(mode), mode = 'round1'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

data = har.loadData(root);
raw = data.trainRaw;
C.Y = categorical(data.trainLabels, data.classes);
C.Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
C.g = groups(1:numel(C.Y));
% 比較の基準は提出用の前処理関数ではなく featurePool から作る（回転の揺れは生の標準偏差）。
% 前処理関数は検証の結果を取り込んで変わるため、通すと基準が変わってしまう。
[~, staticNames] = staticPool(raw(1:2, :, :));
feat = @(w) [ergonomicFeatures(w, false), staticPool(w)];
names = ["erg" + (1:30), staticNames];
C.X = feat(raw);
C.Xte = feat(data.testRaw);
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
    perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
C.Xaug = cellfun(feat, aug, 'UniformOutput', false);
C.shiftNames = ["rotate10", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", "rotate20", "noise0p01"];
sh = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), ...
    perturb(raw, 'rotate', 20, 902), perturb(raw, 'noise', 0.01, 903)};
C.Xshift = cellfun(feat, sh, 'UniformOutput', false);

pick = @(list) arrayfun(@(s) find(names == s), string(list));
base = 1:30;
gyroStd = 14:16;                        % ergonomic32 の gyro_std_x/y/z
keep = setdiff(base, gyroStd, 'stable');
sets = struct();
optsOf = struct();
switch mode
    case 'round1'
        sets = struct();
        sets.base30 = base;
        sets.gyroSway2 = [base, pick(["log_gyro_std_x", "log_gyro_jerk_rms_z"])];
        sets.gyroLog32 = [keep, pick(["log_gyro_std_" + ["x" "y" "z"], "log_gyro_jerk_rms_x", "log_gyro_jerk_rms_z"])];
        sets.gyroBand2 = [base, pick(["log_gyro_x_power_0.3_2hz", "log_gyro_jerk_rms_x"])];
        sets.accJerk2 = [base, pick(["log_jerk_rms_y", "log_jerk_mag_mean"])];
        sets.tiltSway2 = [base, pick(["tilt_y_deg", "log_gyro_std_x"])];
        sets.accJerkOnly = [setdiff(base, 18:22, 'stable'), pick(["log_jerk_rms_" + ["x" "y" "z"], "log_jerk_mag_mean", "log_jerk_mag_max"])];

    case 'round2'
        % 1周目で最良の gyroLog32 を出発点に、残りの列の入れ替えと設定を詰める
        g32 = [keep, pick(["log_gyro_std_" + ["x" "y" "z"], "log_gyro_jerk_rms_x", "log_gyro_jerk_rms_z"])];
        sets.gyroLog32 = g32;
        sets.accStdLog = [setdiff(g32, 5:7, 'stable'), pick("log_acc_std_" + ["x" "y" "z"])];
        sets.swayBandX = [setdiff(g32, pick("log_gyro_jerk_rms_x"), 'stable'), pick("log_gyro_x_power_0.3_2hz")];
        sets.tiltForMag = [setdiff(g32, 17, 'stable'), pick("tilt_y_deg")];
        sets.normForMag = [setdiff(g32, 17, 'stable'), pick("acc_mean_norm")];
        sets.noGyroJerk = [keep, pick("log_gyro_std_" + ["x" "y" "z"])];
        for ks = [0.7, 1.4, 2]
            for box = [10, 30]
                nm = sprintf('gyroLog32_ks%02d_C%d', round(ks * 10), box);
                sets.(nm) = g32;
                optsOf.(nm) = struct('KernelScale', ks * sqrt(32), 'BoxConstraint', box);
            end
        end
        sets.gyroLog32_ks10_C30 = g32;
        optsOf.gyroLog32_ks10_C30 = struct('KernelScale', sqrt(32), 'BoxConstraint', 30);
        % 静止3クラスの躍度の中央値（センサーの雑音の床との比較用）
        Ys = C.Y;
        for c = ["Laying", "Sitting", "Standing"]
            fprintf('median %-8s jerk_rms_xyz %s  gyro_std_x %.4f\n', c, ...
                sprintf('%.4f ', median(C.X(Ys == c, pick("jerk_rms_" + ["x" "y" "z"])))), ...
                10 ^ median(C.X(Ys == c, pick("log_gyro_std_x"))));
        end
end

setNames = string(fieldnames(sets))';
res = cell(numel(setNames), 1);
parfor j = 1:numel(setNames)
    cols = sets.(setNames(j));
    o = struct('KernelScale', sqrt(numel(cols)), 'BoxConstraint', 10);
    if isfield(optsOf, setNames(j)), o = optsOf.(setNames(j)); end
    res{j} = evaluate(C, cols, o);
end
rows = {};
for j = 1:numel(setNames)
    r = res{j};
    rows(end + 1, :) = [{setNames(j), numel(sets.(setNames(j))), r.objective, r.clean, r.cleanSd, ...
        r.test201}, num2cell(r.shift), {r.sitStand}]; %#ok<AGROW>
    fprintf('%-12s p=%2d  J %.4f clean %.4f (sd %.4f) t201 %.4f | %s | sit/stand %d\n', setNames(j), ...
        numel(sets.(setNames(j))), r.objective, r.clean, r.cleanSd, r.test201, sprintf('%.3f ', r.shift), r.sitStand);
end
T = cell2table(rows, 'VariableNames', [{'featureSet', 'nFeatures', 'objective', 'cleanCv', ...
    'cleanCvSd', 'test201'}, cellstr("shift_" + C.shiftNames), {'cvSitStandErrors'}]);
writetable(T, fullfile(out, sprintf('stage5b_six_class_%s.csv', mode)));
fprintf('STAGE5B_DONE\n');
end


function r = evaluate(C, cols, opts)
ids = unique(C.g);
[~, pos] = ismember(C.g, ids);
repeats = 3;
clean = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(C.Xshift));
sitStand = zeros(repeats, 1);
for rep = 1:repeats
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = C.Y;
    for k = 1:5
        tr = fold ~= k;
        [Xtr, Ytr] = trainingRows(C, tr, cols);
        f = fitModel('gauss_svm', Xtr, Ytr, opts);
        h = ~tr;
        pred(h) = f(C.X(h, cols));
        if rep == 1
            for q = 1:numel(C.Xshift)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(C.Xshift{q}(h, cols)) == C.Y(h));
            end
        end
    end
    clean(rep) = mean(pred == C.Y);
    sitStand(rep) = sum(C.Y == 'Sitting' & pred == 'Standing') + sum(C.Y == 'Standing' & pred == 'Sitting');
end
r.clean = mean(clean);
r.cleanSd = std(clean);
r.shift = shiftCorrect / numel(C.Y);
r.objective = mean([r.clean, r.shift(1:5)]);
r.sitStand = mean(sitStand);
[Xall, Yall] = trainingRows(C, true(size(C.Y)), cols);
f = fitModel('gauss_svm', Xall, Yall, opts);
r.test201 = mean(f(C.Xte(:, cols)) == C.Yte);
end


function [X, Y] = trainingRows(C, tr, cols)
X = C.X(tr, cols);
Y = C.Y(tr);
for a = 1:numel(C.Xaug)
    X = [X; C.Xaug{a}(tr, cols)]; %#ok<AGROW>
    Y = [Y; C.Y(tr)]; %#ok<AGROW>
end
end
