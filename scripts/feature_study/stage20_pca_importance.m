function stage20_pca_importance(root)
% 第20段：主成分分析（PCA）で入力を減らせるか、本番32列それぞれの寄与度。
%   物差しは stage18 と同じ。学習用2941窓、記録単位の5分割（乱数 20261116）、
%   学習側だけ拡張8通り、ガウスSVM（KernelScale 1.4√列数、BoxConstraint 10）。
%
%   A. 主成分で減らす（主成分の軸は分割ごとに学習側だけで求める）
%      whiten : 分類学習器の「PCA」と同じ扱い。主成分の得点を標準化してからSVMに渡す
%      keep   : 主成分の得点を大きさのまま渡す。KernelScale は32列と同じ 1.4√32。
%               32成分すべてを使うと、ガウスカーネルは回転で変わらないので本番と一致する
%      比べる相手は、同じ物差しで測った後ろ向き選択の20・12・8・6列（stage13_backward）
%   B. 列ごとの寄与度（4つの方法）
%      permutation : 分割ごとの本番モデルで、評価側の1列だけを行の間で並べ替えたときの
%                    正解率の下がり（10回の平均）。学習し直さない
%      dropColumn  : その列を抜いて学習し直したときの正解率の下がり
%      mrmr        : 分類学習器の特徴量ランキングにある MRMR（fscmrmr）のスコア
%      relieff     : 同じく ReliefF（近傍10）の重み
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
X = productionFeatures(raw);
Xaug = cellfun(@(f) productionFeatures(f(raw)), augFns, 'UniformOutput', false);
names = ["重力の向き x", "重力の向き y", "重力の向き z", "加速度の大きさの平均", ...
    "加速度の標準偏差 x", "加速度の標準偏差 y", "加速度の標準偏差 z", "水平の揺れの平均", ...
    "水平の揺れの標準偏差", "x・y の揺れの標準偏差", "x軸の最大加速度", ...
    "ジャイロxの標準偏差（対数）", "ジャイロyの標準偏差（対数）", "ジャイロzの標準偏差（対数）", ...
    "回転の速さの平均", "躍度 x", "躍度 y", "躍度 z", "上下方向の躍度の最大", "上下方向の躍度の最小", ...
    "上下動の0.5〜3Hzの割合", "水平の揺れ（y）の0.5〜3Hzの割合", "ねじれの0.5〜3Hzの割合", "歩調", ...
    "加速度xとyの相関", "ジャイロyとzの相関", "ジャイロzとxの相関", "上下動の歪度", ...
    "ねじれの1階差分の標準偏差", "ねじれの2階差分の標準偏差", "骨盤の回転の支点までの距離", ...
    "前後の揺れの平均周波数"];
views = ["傾き", "傾き", "傾き", "傾き", repmat("揺れ", 1, 11), repmat("衝撃", 1, 5), ...
    repmat("リズム", 1, 4), repmat("連動", 1, 4), repmat("支点・ねじれ方", 1, 4)];

ids = unique(g);
[~, pos] = ismember(g, ids);
s0 = RandStream('twister', 'Seed', 20261116);
fold = mod(randperm(s0, numel(ids)), 5)' + 1;
fold = fold(pos);

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

%% A. 主成分で減らす
mu0 = mean(X, 1); sd0 = std(X, 0, 1);
[~, ~, ~, ~, explained] = pca((X - mu0) ./ sd0);
B = readtable(fullfile(out, 'stage13_backward.csv'), 'TextType', 'string');
pick = @(n) str2double(split(B.columns(B.nFeatures == n)))';
ks = [2 4 6 8 10 12 16 20 24 32];
tasks = struct('kind', {}, 'k', {}, 'cols', {});
for k = ks
    tasks(end + 1) = struct('kind', "PCA（主成分を標準化）", 'k', k, 'cols', []); %#ok<AGROW>
    tasks(end + 1) = struct('kind', "PCA（主成分の大きさのまま）", 'k', k, 'cols', []); %#ok<AGROW>
end
for n = [32 20 12 8 6]
    if n == 32, c = 1:32; else, c = pick(n); end
    tasks(end + 1) = struct('kind', "列を選ぶ（後ろ向き選択）", 'k', n, 'cols', c); %#ok<AGROW>
end
accA = zeros(numel(tasks), 1);
errA = zeros(numel(tasks), 3);
parfor t = 1:numel(tasks)
    T = tasks(t);
    if isempty(T.cols)
        whiten = startsWith(T.kind, "PCA（主成分を標準化");
        P = pcaCV(X, Xaug, Y, fold, T.k, whiten);
    else
        [~, P] = groupedAcc(X(:, T.cols), cellfun(@(A) A(:, T.cols), Xaug, 'UniformOutput', false), Y, fold);
    end
    accA(t) = mean(P == Y);
    errA(t, :) = errorTypes(P, Y, data.classes);
end
rows = {};
for t = 1:numel(tasks)
    cum = NaN;
    if startsWith(tasks(t).kind, "PCA"), cum = sum(explained(1:tasks(t).k)); end
    rows(end + 1, :) = {tasks(t).kind, tasks(t).k, accA(t), cum, errA(t, 1), errA(t, 2), errA(t, 3)}; %#ok<AGROW>
    fprintf('%-26s %2d  %.4f  寄与率累計 %5.1f%%  誤り 動作の間 %d 座位↔立位 %d その他 %d\n', ...
        tasks(t).kind, tasks(t).k, accA(t), cum, errA(t, :));
end
TA = cell2table(rows, 'VariableNames', {'method', 'nInputs', 'groupedCv', 'pcaCumExplained', ...
    'errDynamic', 'errSitStand', 'errOther'});
writetable(TA, fullfile(out, 'stage20_pca.csv'), 'Encoding', 'UTF-8');

%% B. 列ごとの寄与度
% permutation：分割ごとに本番の設定で学習し、評価側の1列を並べ替える
nPerm = 10;
permCorrect = zeros(5, 32);
permErr = zeros(5, 32, 3);
baseCorrect = zeros(5, 1);
baseErr = zeros(5, 3);
parfor k = 1:5
    tr = fold ~= k; te = find(~tr);
    Xtr = [X(tr, :); cell2mat(cellfun(@(A) A(tr, :), Xaug(:), 'UniformOutput', false))];
    f = fitModel('gauss_svm', Xtr, repmat(Y(tr), numel(Xaug) + 1, 1), ...
        struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(32)));
    Xte = X(te, :); Yte = Y(te);
    p0 = f(Xte);
    baseCorrect(k) = sum(p0 == Yte);
    baseErr(k, :) = errorTypes(p0, Yte, data.classes);
    s = RandStream('twister', 'Seed', 20261127 + k);
    pc = zeros(1, 32); pe = zeros(32, 3);
    for j = 1:32
        for r = 1:nPerm
            Xp = Xte;
            Xp(:, j) = Xte(randperm(s, numel(te)), j);
            p = f(Xp);
            pc(j) = pc(j) + sum(p == Yte) / nPerm;
            pe(j, :) = pe(j, :) + errorTypes(p, Yte, data.classes) / nPerm;
        end
    end
    permCorrect(k, :) = pc;
    permErr(k, :, :) = reshape(pe, 1, 32, 3);
end
N = numel(Y);
fullAcc = sum(baseCorrect) / N;
fullErr = sum(baseErr, 1);
permDrop = fullAcc - sum(permCorrect, 1) / N;
permErrUp = squeeze(sum(permErr, 1)) - fullErr;      % 32×3

% dropColumn：その列を抜いて学習し直す
dropAcc = zeros(1, 32);
dropErr = zeros(32, 3);
parfor j = 1:32
    c = setdiff(1:32, j);
    [a, P] = groupedAcc(X(:, c), cellfun(@(A) A(:, c), Xaug, 'UniformOutput', false), Y, fold);
    dropAcc(j) = a;
    dropErr(j, :) = errorTypes(P, Y, data.classes);
end
dropDrop = fullAcc - dropAcc;
dropErrUp = dropErr - fullErr;

% 分類学習器の特徴量ランキングと同じ2つ（学習用2941窓、拡張なし）
[mrmrIdx, mrmrScore] = fscmrmr(X, Y);
mrmrRank = zeros(1, 32); mrmrRank(mrmrIdx) = 1:32;
[reliefIdx, reliefW] = relieff((X - mu0) ./ sd0, Y, 10);
reliefRank = zeros(1, 32); reliefRank(reliefIdx) = 1:32;

rk = @(v) tiedrank(-v);
permRank = rk(permDrop); dropRank = rk(dropDrop);
meanRank = mean([permRank; dropRank; mrmrRank; reliefRank], 1);
TB = table((1:32)', names', views', permDrop' * 100, permRank', permErrUp(:, 1), permErrUp(:, 2), ...
    dropDrop' * 100, dropRank', dropErrUp(:, 1), dropErrUp(:, 2), mrmrScore', mrmrRank', ...
    reliefW', reliefRank', meanRank', ...
    'VariableNames', {'column', 'feature', 'view', 'permDropPt', 'permRank', 'permErrDynamicUp', ...
    'permErrSitStandUp', 'dropDropPt', 'dropRank', 'dropErrDynamicUp', 'dropErrSitStandUp', ...
    'mrmrScore', 'mrmrRank', 'relieffWeight', 'relieffRank', 'meanRank'});
TB = sortrows(TB, 'meanRank');
writetable(TB, fullfile(out, 'stage20_importance.csv'), 'Encoding', 'UTF-8');
fprintf('\n32列 %.4f  誤り 動作の間 %d 座位↔立位 %d その他 %d\n', fullAcc, fullErr);
disp(TB(:, {'column', 'feature', 'permDropPt', 'permRank', 'dropDropPt', 'dropRank', 'mrmrRank', 'relieffRank', 'meanRank'}));
fprintf('STAGE20_DONE\n');
end


function P = pcaCV(X, Xaug, Y, fold, k, whiten)
% 主成分の軸と標準化の定数は、分割ごとに学習側（拡張を含む）だけで求める。
P = Y;
for f = 1:5
    tr = fold ~= f;
    Xtr = [X(tr, :); cell2mat(cellfun(@(A) A(tr, :), Xaug(:), 'UniformOutput', false))];
    Ytr = repmat(Y(tr), numel(Xaug) + 1, 1);
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-12) = 1;
    Ztr = (Xtr - mu) ./ sd;
    [coeff, ~, ~, ~, ~, c0] = pca(Ztr);
    W = coeff(:, 1:k);
    Str = (Ztr - c0) * W;
    Ste = (((X(~tr, :) - mu) ./ sd) - c0) * W;
    if whiten
        g = fitModel('gauss_svm', Str, Ytr, struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(k)));
        P(~tr) = g(Ste);
    else
        t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 1.4 * sqrt(size(X, 2)), ...
            'BoxConstraint', 10);
        mdl = fitcecoc(Str, Ytr, 'Learners', t, 'Coding', 'onevsone');
        P(~tr) = predict(mdl, Ste);
    end
end
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


function e = errorTypes(P, Y, classes)
% 誤りを3種類に分ける：歩行・上り・下りの間、座位↔立位、それ以外
dynC = categorical(["Walking", "WalkingUpstairs", "WalkingDownstairs"], classes);
ssC = categorical(["Sitting", "Standing"], classes);
wrong = P ~= Y;
dyn = ismember(Y, dynC) & ismember(P, dynC);
ss = ismember(Y, ssC) & ismember(P, ssC);
e = [sum(wrong & dyn), sum(wrong & ss), sum(wrong & ~dyn & ~ss)];
end
