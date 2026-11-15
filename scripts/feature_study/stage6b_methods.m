function stage6b_methods(root)
% 第6段の2：本番の30列（対数＋全体で1回の標準化）・拡張8通り・ずれの評価をそろえて、
% 手法だけを変えて比べる。特徴量は stage6_production_loop が作ったキャッシュを使う。
% 標準化は fitModel が学習側の全クラスで1回だけ行う（本番の固定定数と同じ考え方）。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
load(fullfile(out, 'stage6_cache.mat'), 'C');
if isempty(gcp('nocreate')), parpool('Processes', 8); end
cols = 1:30;
jobs = {'gauss_svm', struct('KernelScale', 1.4 * sqrt(30), 'BoxConstraint', 10); ...
    'nnet', struct(); 'bagged_tree', struct(); 'subspace_knn', struct(); ...
    'knn', struct(); 'linear_svm', struct()};
res = cell(size(jobs, 1), 1);
parfor j = 1:size(jobs, 1)
    res{j} = evaluate(C, cols, jobs{j, 1}, jobs{j, 2});
end
rows = {};
for j = 1:size(jobs, 1)
    r = res{j};
    rows(end + 1, :) = [{jobs{j, 1}, r.objective, r.clean, r.cleanSd, r.worst, r.test201}, ...
        num2cell(r.shift), {r.sitStand}]; %#ok<AGROW>
    fprintf('%-13s J %.4f clean %.4f (sd %.4f, worst %.4f) t201 %.4f | %s | sit/stand %.1f\n', ...
        jobs{j, 1}, r.objective, r.clean, r.cleanSd, r.worst, r.test201, sprintf('%.3f ', r.shift), r.sitStand);
end
T = cell2table(rows, 'VariableNames', [{'method', 'objective', 'cleanCv', 'cleanCvSd', ...
    'cleanCvWorst', 'test201'}, cellstr("shift_" + C.shiftNames), {'cvSitStandErrors'}]);
writetable(T, fullfile(out, 'stage6b_methods.csv'));
fprintf('STAGE6B_DONE\n');
end


function r = evaluate(C, cols, method, opts)
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
        [X, Y] = rowsOf(C, tr, cols);
        rng(rep * 100 + k, 'twister');
        f = fitModel(method, X, Y, opts);
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
[X, Y] = rowsOf(C, true(size(C.Y)), cols);
rng(7, 'twister');
f = fitModel(method, X, Y, opts);
r.test201 = mean(f(C.Xte(:, cols)) == C.Yte);
end


function [X, Y] = rowsOf(C, tr, cols)
X = C.X(tr, cols);
Y = C.Y(tr);
for a = 1:numel(C.Xaug)
    X = [X; C.Xaug{a}(tr, cols)]; %#ok<AGROW>
    Y = [Y; C.Y(tr)]; %#ok<AGROW>
end
end
