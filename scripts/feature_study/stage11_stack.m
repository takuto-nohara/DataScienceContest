function stage11_stack(root, baseCols)
% 第11段：座位と立位の「専門家スコア」を1列にして、本番の列に足す。
%   座位と立位の窓だけで、正則化したロジスティック回帰（線形）を学習する。その出力
%   （重み×標準化した候補列＋切片）を1列の特徴量にする。前処理関数では固定の定数
%   （候補列の平均・標準偏差、重み、切片）で計算できるので、試行ごとに独立に求まる。
%   候補列は、本番の34列、bioPool の29列、staticPool の42列。
% 交差検証では、専門家も分割ごとに学習側の窓（拡張を含む）だけで学習する。
%   baseCols : 本番の34列のうち、6クラスのSVMに使う列（専門家スコアはこの後ろに足す）
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(baseCols), baseCols = 1:32; end
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
feat = @(w) [productionFeatures(w), bioPool(w), staticPool(w)];
X = feat(raw);
Xaug = cellfun(feat, aug, 'UniformOutput', false);
Xshift = cellfun(feat, shifts, 'UniformOutput', false);
pool = 1:size(X, 2);
small = 1:34;
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

% 専門家なし／候補34列の専門家／候補105列の専門家（正則化の強さ3通り）
cfg = struct('label', {"(base)", "expert34 lambda1e-4", "expert34 lambda1e-3", ...
    "expertAll lambda1e-4", "expertAll lambda1e-3", "expertAll lambda1e-2"}, ...
    'expertCols', {[], small, small, pool, pool, pool}, ...
    'lambda', {NaN, 1e-4, 1e-3, 1e-4, 1e-3, 1e-2});
res = cell(numel(cfg), 1);
parfor j = 1:numel(cfg)
    res{j} = groupedStack(X, Xaug, Y, g, Xshift, baseCols, cfg(j).expertCols, cfg(j).lambda, 20261115, 3);
end
rows = {};
for j = 1:numel(cfg)
    r = res{j};
    rows(end + 1, :) = [{cfg(j).label, r.acc, r.accSd, r.objective, r.sitStand, r.expertAcc}, num2cell(r.shift)]; %#ok<AGROW>
    fprintf('%-22s cv %.4f (sd %.4f) J %.4f sit/stand %.1f expert-alone %.4f\n', cfg(j).label, ...
        r.acc, r.accSd, r.objective, r.sitStand, r.expertAcc);
end
T = cell2table(rows, 'VariableNames', [{'variant', 'groupedCv', 'groupedCvSd', 'objective', ...
    'sitStandErrors', 'expertSitStandAcc'}, cellstr("shift_" + shiftNames)]);
writetable(T, fullfile(out, 'stage11_stack.csv'));
disp(T);
fprintf('STAGE11_DONE\n');
end


function r = groupedStack(X, Xaug, Y, g, Xshift, baseCols, expertCols, lambda, seedBase, repeats)
ids = unique(g);
[~, pos] = ismember(g, ids);
acc = zeros(repeats, 1); ss = zeros(repeats, 1); ea = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(Xshift));
useExpert = ~isempty(expertCols);
p = numel(baseCols) + useExpert;
opts = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(p));
for rep = 1:repeats
    s = RandStream('twister', 'Seed', seedBase + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    expertCorrect = 0; expertTotal = 0;
    for k = 1:5
        tr = fold ~= k;
        Xtr = X(tr, :); Ytr = Y(tr);
        for a = 1:numel(Xaug)
            Xtr = [Xtr; Xaug{a}(tr, :)]; %#ok<AGROW>
            Ytr = [Ytr; Y(tr)]; %#ok<AGROW>
        end
        build = @(A) A(:, baseCols);
        if useExpert
            score = trainExpert(Xtr(:, expertCols), Ytr, lambda);
            build = @(A) [A(:, baseCols), score(A(:, expertCols))];
            te = ~tr & ismember(Y, categorical(["Sitting", "Standing"]));
            sc = score(X(te, expertCols));
            expertCorrect = expertCorrect + sum((sc > 0) == (Y(te) == 'Standing'));
            expertTotal = expertTotal + sum(te);
        end
        f = fitModel('gauss_svm', build(Xtr), Ytr, opts);
        pred(~tr) = f(build(X(~tr, :)));
        if rep == 1
            for q = 1:numel(Xshift)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(build(Xshift{q}(~tr, :))) == Y(~tr));
            end
        end
    end
    acc(rep) = mean(pred == Y);
    ss(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
    ea(rep) = expertCorrect / max(expertTotal, 1);
end
r.acc = mean(acc);
r.accSd = std(acc);
r.sitStand = mean(ss);
r.expertAcc = mean(ea);
r.shift = shiftCorrect / numel(Y);
r.objective = mean([r.acc, r.shift([1 3 4 5 6])]);
end


function score = trainExpert(X, Y, lambda)
% 座位(0)と立位(1)のロジスティック回帰。標準化の定数と重みを固定して返す。
rows = Y == 'Sitting' | Y == 'Standing';
Xs = X(rows, :);
mu = mean(Xs, 1);
sd = std(Xs, 0, 1);
sd(sd < 1e-12) = 1;
mdl = fitclinear((Xs - mu) ./ sd, double(Y(rows) == 'Standing'), 'Learner', 'logistic', ...
    'Regularization', 'ridge', 'Lambda', lambda, 'Solver', 'lbfgs');
w = mdl.Beta; b = mdl.Bias;
score = @(A) ((A - mu) ./ sd) * w + b;
end
