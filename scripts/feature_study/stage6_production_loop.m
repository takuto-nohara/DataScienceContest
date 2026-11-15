function stage6_production_loop(root, mode, setName, standardize)
% 第6段：本番と同じ学習方法（fitcecoc＋templateSVM、分類学習器と同じ）で比べるループ。
%   mode = 'sets' : 空き2列の使い道 × 標準化の方式
%          'grid' : 選んだ組（setName, standardize）でカーネルスケールとボックス制約
%   標準化の方式
%     'pair'   : templateSVM の Standardize=true（2クラスの組ごとに標準化。分類学習器の既定）
%     'global' : 学習側の全クラスで1回だけ標準化し、Standardize=false。本番では学習データの
%                平均と標準偏差を前処理関数に定数として書き、分類学習器の標準化を切る
% 目的関数 J はずれなしと現実的なずれ5種の正解率の平均（第4段と同じ）。テスト201窓は選択に使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(mode), mode = 'sets'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here, fullfile(root, 'submission', 'ergonomic32_svm'));
out = fullfile(root, 'evaluation', 'feature_study');
cacheFile = fullfile(out, 'stage6_cache.mat');
if isfile(cacheFile)
    load(cacheFile, 'C');
else
    C = buildCache(root);
    save(cacheFile, 'C', '-v7.3');
end
if isempty(gcp('nocreate')), parpool('Processes', 8); end

pick = @(list) arrayfun(@(s) find(C.names == s), string(list));
base = 1:30;
sets = struct();
sets.log30 = base;
sets.normTilt = [base, pick(["acc_mean_norm", "tilt_y_deg"])];
sets.tiltYZ = [base, pick(["tilt_y_deg", "tilt_z_deg"])];
sets.swayExtra = [base, pick(["log_gyro_x_power_0.3_2hz", "log_gyro_jerk_rms_z"])];
sets.normForMag = [setdiff(base, 17, 'stable'), pick("acc_mean_norm")];

jobs = {};
switch mode
    case 'sets'
        for s = string(fieldnames(sets))'
            for st = ["pair", "global"]
                p = numel(sets.(s));
                jobs(end + 1, :) = {s, st, sqrt(p), 10}; %#ok<AGROW>
            end
        end
    case 'grid'
        p = numel(sets.(setName));
        for ks = [0.7, 1, 1.4, 2]
            for box = [3, 10, 30]
                jobs(end + 1, :) = {string(setName), string(standardize), ks * sqrt(p), box}; %#ok<AGROW>
            end
        end
end

res = cell(size(jobs, 1), 1);
parfor j = 1:size(jobs, 1)
    res{j} = evaluate(C, sets.(jobs{j, 1}), jobs{j, 2}, jobs{j, 3}, jobs{j, 4});
end
rows = {};
for j = 1:size(jobs, 1)
    r = res{j};
    p = numel(sets.(jobs{j, 1}));
    rows(end + 1, :) = [{jobs{j, 1}, jobs{j, 2}, p, jobs{j, 3} / sqrt(p), jobs{j, 4}, r.objective, ...
        r.clean, r.cleanSd, r.worst, r.test201}, num2cell(r.shift), {r.sitStand}]; %#ok<AGROW>
    fprintf('%-11s %-6s p=%2d ks %.2f C %4.1f  J %.4f clean %.4f (sd %.4f, worst %.4f) t201 %.4f | %s | sit/stand %.1f\n', ...
        jobs{j, 1}, jobs{j, 2}, p, jobs{j, 3} / sqrt(p), jobs{j, 4}, r.objective, r.clean, r.cleanSd, ...
        r.worst, r.test201, sprintf('%.3f ', r.shift), r.sitStand);
end
T = cell2table(rows, 'VariableNames', [{'featureSet', 'standardize', 'nFeatures', ...
    'kernelScaleOverSqrtP', 'boxConstraint', 'objective', 'cleanCv', 'cleanCvSd', 'cleanCvWorst', ...
    'test201'}, cellstr("shift_" + C.shiftNames), {'cvSitStandErrors'}]);
writetable(T, fullfile(out, sprintf('stage6_%s.csv', mode)));
fprintf('STAGE6_DONE\n');
end


function C = buildCache(root)
data = har.loadData(root);
raw = data.trainRaw;
C.Y = categorical(data.trainLabels, data.classes);
C.Yte = categorical(data.testLabels, data.classes);
C.classes = categorical(data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
C.g = groups(1:numel(C.Y));
pre = @(w) Pretreatment_MATLABcontest_Advanced(w(:, :, 1), w(:, :, 2), w(:, :, 3), ...
    w(:, :, 4), w(:, :, 5), w(:, :, 6));
[~, staticNames] = staticPool(raw(1:2, :, :));
feat = @(w) [subsref(pre(w), substruct('()', {':', 1:30})), staticPool(w)];
C.names = ["erg" + (1:30), staticNames];
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
end


function r = evaluate(C, cols, standardize, kernelScale, box)
ids = unique(C.g);
[~, pos] = ismember(C.g, ids);
repeats = 3;
clean = zeros(repeats, 1);
sitStand = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(C.Xshift));
for rep = 1:repeats
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = C.Y;
    for k = 1:5
        tr = fold ~= k;
        f = train(C, tr, cols, standardize, kernelScale, box);
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
r.worst = min(clean);
r.sitStand = mean(sitStand);
r.shift = shiftCorrect / numel(C.Y);
r.objective = mean([r.clean, r.shift(1:5)]);
f = train(C, true(size(C.Y)), cols, standardize, kernelScale, box);
r.test201 = mean(f(C.Xte(:, cols)) == C.Yte);
end


function f = train(C, tr, cols, standardize, kernelScale, box)
X = C.X(tr, cols);
Y = C.Y(tr);
for a = 1:numel(C.Xaug)
    X = [X; C.Xaug{a}(tr, cols)]; %#ok<AGROW>
    Y = [Y; C.Y(tr)]; %#ok<AGROW>
end
if standardize == "global"
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma < 1e-12) = 1;
    scale = @(Z) (Z - mu) ./ sigma;
    X = scale(X);
else
    scale = @(Z) Z;
end
m = fitcecoc(X, Y, 'Coding', 'onevsone', 'ClassNames', C.classes, 'Learners', templateSVM( ...
    'KernelFunction', 'gaussian', 'KernelScale', kernelScale, 'BoxConstraint', box, ...
    'Standardize', standardize == "pair"));
f = @(Z) predict(m, scale(Z));
end
