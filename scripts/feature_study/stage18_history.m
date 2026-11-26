function stage18_history(root)
% 第18段：特徴量の試行錯誤の5段階を同じ物差しで測り直し、正解率を観点ごとに分ける。
%   5段階
%   A 主催者の12列（6軸の平均と標準偏差）
%   B 人体工学30列（回転の揺れは生の標準偏差）
%   C Bの回転の揺れ3列を常用対数にする
%   D Cにジャイロx（鉛直軸まわり）の1階差分・2階差分の標準偏差を足す（32列）
%   E 本番の32列（Dから acc_mag_std と vertical_std を外し、支点と前後の揺れの周波数を足す）
%   物差しは stage15 と同じ。学習用2941窓、記録単位の5分割（乱数 20261116）、
%   学習側だけ拡張8通り、ガウスSVM（KernelScale 1.4√列数、BoxConstraint 10）。
%   観点ごとの内訳はシャープレイ値で求める。観点の組すべて（2^観点の数）で正解率を測り、
%   各観点を足したときの増え方を、足す順番すべてで平均する。内訳の合計は、
%   その段階の正解率から「列なし」（最も多い行動を答える）の正解率を引いた値に一致する。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
g = recordingGroups(cat(1, raw, data.testRaw));
g = g(1:numel(Y));
augFns = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};

views = ["傾き", "揺れ", "衝撃", "リズム", "連動", "支点・ねじれ方", "ジャイロの平均"];
ergoGroups = [1 1 1 1, 2 2 2 2 2 2 2 2 2 2 2 2 2, 3 3 3 3 3, 4 4 4 4, 5 5 5 5];
prodGroups = [1 1 1 1, 2 2 2 2 2 2 2 2 2 2 2, 3 3 3 3 3, 4 4 4 4, 5 5 5 5, 6 6 6 6];
stages = struct('name', {"主催者の12列", "観点で30列", "揺れを対数に", "ねじれの細かさ", "回転の支点"}, ...
    'f', {@organizer12, @(w) ergonomicFeatures(w, false), @(w) ergonomicFeatures(w, true), @stageD, @productionFeatures}, ...
    'groups', {[1 1 1 2 2 2 7 7 7 2 2 2], ergoGroups, ergoGroups, [ergoGroups, 6 6], prodGroups});

ids = unique(g);
[~, pos] = ismember(g, ids);
s0 = RandStream('twister', 'Seed', 20261116);
fold = mod(randperm(s0, numel(ids)), 5)' + 1;
fold = fold(pos);
base = 0;
for k = 1:5
    tr = fold ~= k;
    cnt = countcats(Y(tr));
    [~, m] = max(cnt);
    base = base + sum(Y(~tr) == categorical(data.classes(m), data.classes));
end
base = base / numel(Y);
fprintf('列なし %.4f\n', base);

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

rows = {};
pairRows = {};
for s = 1:numel(stages)
    X = stages(s).f(raw);
    Xaug = cellfun(@(f) stages(s).f(f(raw)), augFns, 'UniformOutput', false);
    grp = stages(s).groups;
    gu = unique(grp, 'stable');
    nG = numel(gu);
    masks = dec2bin(1:2 ^ nG - 1) == '1';     % 空でない組すべて
    acc = zeros(size(masks, 1), 1);
    preds = cell(size(masks, 1), 1);
    parfor m = 1:size(masks, 1)
        cols = find(ismember(grp, gu(masks(m, :))));
        [acc(m), preds{m}] = groupedAcc(X(:, cols), cellfun(@(A) A(:, cols), Xaug, 'UniformOutput', false), Y, fold);
    end
    v = @(mask) valueOf(mask, masks, acc, base);
    phi = zeros(1, nG);
    for i = 1:nG
        others = setdiff(1:nG, i);
        for r = 0:numel(others)
            if r == 0, combos = zeros(1, 0); else, combos = nchoosek(others, r); end
            w = factorial(r) * factorial(nG - r - 1) / factorial(nG);
            for c = 1:size(combos, 1)
                S = false(1, nG); S(combos(c, :)) = true;
                Si = S; Si(i) = true;
                phi(i) = phi(i) + w * (v(Si) - v(S));
            end
        end
    end
    full = acc(end);
    P = preds{end};
    fprintf('%s  %d列  %.4f  内訳合計 %.4f\n', stages(s).name, numel(grp), full, base + sum(phi));
    for i = 1:nG
        fprintf('   %-12s %+.4f\n', views(gu(i)), phi(i));
        rows(end + 1, :) = {stages(s).name, numel(grp), full, base, views(gu(i)), sum(grp == gu(i)), phi(i)}; %#ok<AGROW>
    end
    % 行動の組ごとの誤り（どの取り違えが減ったか）
    pairs = {["Walking", "WalkingUpstairs", "WalkingDownstairs"], ["Sitting", "Standing"], "Laying"};
    pairNames = ["歩行・上り・下りの間", "座位↔立位", "静止と動作の間・臥位"];
    dyn = ismember(Y, categorical(pairs{1}, data.classes)) & ismember(P, categorical(pairs{1}, data.classes));
    ss = ismember(Y, categorical(pairs{2}, data.classes)) & ismember(P, categorical(pairs{2}, data.classes));
    wrong = P ~= Y;
    e = [sum(wrong & dyn), sum(wrong & ss), sum(wrong & ~dyn & ~ss)];
    for q = 1:3
        pairRows(end + 1, :) = {stages(s).name, pairNames(q), e(q)}; %#ok<AGROW>
    end
    fprintf('   誤り：動作の間 %d  座位↔立位 %d  その他 %d\n', e);
end
T = cell2table(rows, 'VariableNames', {'stage', 'nFeatures', 'groupedCv', 'noFeature', 'view', 'nCols', 'shapley'});
writetable(T, fullfile(out, 'stage18_history_shapley.csv'), 'Encoding', 'UTF-8');
T2 = cell2table(pairRows, 'VariableNames', {'stage', 'errorType', 'errors'});
writetable(T2, fullfile(out, 'stage18_history_errors.csv'), 'Encoding', 'UTF-8');
fprintf('STAGE18_DONE\n');
end


function x = valueOf(mask, masks, acc, base)
if ~any(mask), x = base; return, end
x = acc(all(masks == mask, 2));
end


function [a, pred] = groupedAcc(X, Xaug, Y, fold)
p = size(X, 2);
opts = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(p));
pred = Y;
for k = 1:5
    tr = fold ~= k;
    Xtr = [X(tr, :); cell2mat(cellfun(@(A) A(tr, :), Xaug(:), 'UniformOutput', false))];
    f = fitModel('gauss_svm', Xtr, repmat(Y(tr), numel(Xaug) + 1, 1), opts);
    pred(~tr) = f(X(~tr, :));
end
a = mean(pred == Y);
end


function X = organizer12(raw)
m = squeeze(mean(raw, 2));
s = squeeze(std(raw, 0, 2));
X = [m(:, 1:3), s(:, 1:3), m(:, 4:6), s(:, 4:6)];
end


function F = stageD(raw)
A = araiDiff24(raw);
F = [ergonomicFeatures(raw, true), A(:, [10, 22])];
end
