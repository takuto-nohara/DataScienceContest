function stage15_model_compare(root)
% 第15段：発表用に、同じ条件で学習器と特徴量を比べ、混同行列を残す。
%   特徴量 : 本番の32列（productionFeatures）と、主催者の雛形の12列（各軸の平均と標準偏差）
%   学習器 : ガウスSVM（本番）、ニューラルネット、バギング木、k-NN、線形SVM、決定木
%   学習   : 学習用2941窓＋拡張8通り（本番と同じ）
%   測る値 : 学習用の記録単位の5分割交差検証（1回）、検証用407窓、テスト201窓、
%            検証用に装着角度20°のずれを加えた407窓、検証用の混同行列
% テスト201窓と検証用407窓は、どの選択にも使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
ver = har.readVerification(root);
cls = categorical(data.classes);
Ytr = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
Yv = categorical(ver.labels, data.classes);
g = recordingGroups(cat(1, data.trainRaw, data.testRaw));
g = g(1:numel(Ytr));
augFns = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
rot = @(w) perturb(w, 'rotate', 20, 902);
featureSets = {'ergonomic32', @productionFeatures; 'organizer12', @organizer12};
models = {'ergonomic32', 'gauss_svm'; 'ergonomic32', 'nnet'; 'ergonomic32', 'bagged_tree'; ...
    'ergonomic32', 'knn'; 'ergonomic32', 'linear_svm'; 'ergonomic32', 'tree'; 'organizer12', 'gauss_svm'};

F = struct();
for s = 1:size(featureSets, 1)
    f = featureSets{s, 2};
    F.(featureSets{s, 1}) = struct('X', f(data.trainRaw), 'Xte', f(data.testRaw), 'Xv', f(ver.raw), ...
        'Xvr', f(rot(ver.raw)), 'Xaug', {cellfun(@(a) f(a(data.trainRaw)), augFns, 'UniformOutput', false)});
end
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));
ids = unique(g);
[~, pos] = ismember(g, ids);
s0 = RandStream('twister', 'Seed', 20261116);
fold = mod(randperm(s0, numel(ids)), 5)' + 1;
fold = fold(pos);
res = cell(size(models, 1), 1);
parfor m = 1:size(models, 1)
    D = F.(models{m, 1});
    p = size(D.X, 2);
    opts = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(p));
    pred = Ytr;
    for k = 1:5
        tr = fold ~= k;
        Xtr = [D.X(tr, :); cell2mat(cellfun(@(A) A(tr, :), D.Xaug(:), 'UniformOutput', false))];
        f = fitModel(models{m, 2}, Xtr, repmat(Ytr(tr), 9, 1), opts);
        pred(~tr) = f(D.X(~tr, :));
    end
    f = fitModel(models{m, 2}, [D.X; cell2mat(D.Xaug(:))], repmat(Ytr, 9, 1), opts);
    r = struct();
    r.cv = mean(pred == Ytr);
    r.cvSitStand = sum(ismember(Ytr, cls(4:5)) & ismember(pred, cls(4:5)) & pred ~= Ytr);
    r.pv = f(D.Xv);
    r.test = sum(f(D.Xte) == Yte);
    r.rot = mean(f(D.Xvr) == Yv);
    res{m} = r;
end
rows = {};
for m = 1:size(models, 1)
    r = res{m};
    ok = r.pv == Yv;
    ss = sum(ismember(Yv, cls(4:5)) & ismember(r.pv, cls(4:5)) & ~ok);
    rows(end + 1, :) = {models{m, 1}, models{m, 2}, r.cv, r.cvSitStand, sum(ok), mean(ok), ss, r.test, r.rot}; %#ok<AGROW>
    cm = confusionmat(Yv, r.pv, 'Order', cls);
    writetable(array2table(cm, 'VariableNames', cellstr(data.classes), 'RowNames', cellstr(data.classes)), ...
        fullfile(out, sprintf('stage15_confusion_%s_%s.csv', models{m, 1}, models{m, 2})), 'WriteRowNames', true);
    fprintf('%-12s %-12s cv %.4f (sit/stand %d)  verification %d/407 (%.4f, sit/stand %d)  test %d/201  rot20 %.4f\n', ...
        models{m, 1}, models{m, 2}, r.cv, r.cvSitStand, sum(ok), mean(ok), ss, r.test, r.rot);
end
T = cell2table(rows, 'VariableNames', {'features', 'method', 'groupedCv1', 'cvSitStandErrors', ...
    'verCorrect', 'verAcc', 'verSitStandErrors', 'testCorrect', 'verRotate20'});
writetable(T, fullfile(out, 'stage15_model_compare.csv'));
fprintf('STAGE15_DONE\n');
end


function X = organizer12(raw)
% 主催者の雛形の前処理関数と同じ12列（各軸の平均と標準偏差）
m = squeeze(mean(raw, 2));
s = squeeze(std(raw, 0, 2));
X = [m(:, 1:3), s(:, 1:3), m(:, 4:6), s(:, 4:6)];
end
