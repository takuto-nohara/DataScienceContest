function stage10_budget(root, mode, drops)
% 第10段：特徴量は32個までという規則に合わせる。
% 本番の34列（productionFeatures）から2列を外して32列にする。
%   mode = 'dropone' : 1〜32列目を1列ずつ外した33列を、記録単位の交差検証（3回）で測る
%   mode = 'confirm' : drops（外す列番号の組の cell）を、分割の乱数を変えた5回で確かめる
% 拡張8通り、ガウスSVM（カーネルスケール 1.4×√列数、ボックス制約 10）、全体で1回の標準化。
% テスト201窓は使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(mode), mode = 'dropone'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
g = groups(1:numel(Y));
shiftNames = ["rotate10", "rotate20", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", "noise0p01"];
shifts = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'rotate', 20, 902), ...
    perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), perturb(raw, 'noise', 0.01, 903)};
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
    perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
names = productionNames(raw);
X = productionFeatures(raw);
Xaug = cellfun(@productionFeatures, aug, 'UniformOutput', false);
Xshift = cellfun(@productionFeatures, shifts, 'UniformOutput', false);
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

switch mode
    case 'dropone'
        jobs = [{1:34}, arrayfun(@(c) setdiff(1:34, c), 1:32, 'UniformOutput', false)];
        labels = ["(all 34)", "-" + names(1:32)];
        seedBase = 20261115; repeats = 3; file = 'stage10_dropone.csv';
    case 'confirm'
        jobs = [{1:32}, cellfun(@(d) setdiff(1:34, d), drops, 'UniformOutput', false)];
        labels = ["(32 before stage9)", cellfun(@(d) "-" + strjoin(names(d), " -"), drops)];
        seedBase = 20271115; repeats = 5; file = 'stage10_confirm.csv';
end
res = cell(numel(jobs), 1);
parfor j = 1:numel(jobs)
    cols = jobs{j};
    o = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(numel(cols)));
    res{j} = groupedAug(X(:, cols), cellfun(@(A) A(:, cols), Xaug, 'UniformOutput', false), Y, g, ...
        cellfun(@(A) A(:, cols), Xshift, 'UniformOutput', false), o, seedBase, repeats);
end
rows = {};
for j = 1:numel(jobs)
    r = res{j};
    rows(end + 1, :) = [{labels(j), numel(jobs{j}), r.acc, r.accSd, r.objective, r.sitStand}, num2cell(r.shift)]; %#ok<AGROW>
    fprintf('%-45s cv %.4f (sd %.4f) J %.4f sit/stand %.1f\n', labels(j), r.acc, r.accSd, r.objective, r.sitStand);
end
T = cell2table(rows, 'VariableNames', [{'variant', 'nFeatures', 'groupedCv', 'groupedCvSd', 'objective', ...
    'sitStandErrors'}, cellstr("shift_" + shiftNames)]);
T = sortrows(T, 'objective', 'descend');
writetable(T, fullfile(out, file));
disp(T);
fprintf('STAGE10_DONE\n');
end


function names = productionNames(raw)
[~, ergo] = ergonomicFeatures(raw(1:2, :, :), true);
names = [ergo, "gyro_x_diff_std", "gyro_x_diff2_std", "pivot_height_mean", "sway_fmean_ap"];
end


function r = groupedAug(X, Xaug, Y, g, Xshift, opts, seedBase, repeats)
% 記録単位の5分割を repeats 回。学習側だけ拡張する。1回目でずれも測る。
ids = unique(g);
[~, pos] = ismember(g, ids);
acc = zeros(repeats, 1); ss = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(Xshift));
for rep = 1:repeats
    s = RandStream('twister', 'Seed', seedBase + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        tr = fold ~= k;
        Xtr = X(tr, :); Ytr = Y(tr);
        for a = 1:numel(Xaug)
            Xtr = [Xtr; Xaug{a}(tr, :)]; %#ok<AGROW>
            Ytr = [Ytr; Y(tr)]; %#ok<AGROW>
        end
        f = fitModel('gauss_svm', Xtr, Ytr, opts);
        pred(~tr) = f(X(~tr, :));
        if rep == 1
            for q = 1:numel(Xshift)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(Xshift{q}(~tr, :)) == Y(~tr));
            end
        end
    end
    acc(rep) = mean(pred == Y);
    ss(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
r.acc = mean(acc);
r.accSd = std(acc);
r.sitStand = mean(ss);
r.shift = shiftCorrect / numel(Y);
r.objective = mean([r.acc, r.shift([1 3 4 5 6])]);
end
