function stage19_reliability(root)
% 第19段：列を減らした組の正解率が、選び方の水増しなしでも出るかを確かめる。
%   第13段の後ろ向き選択は、学習用2941窓の交差検証で列を選び、同じ窓で測った。
%   選ぶときに評価側の窓も見ているので、値は高めに出るおそれがある。そこで3通りで測る。
%   1 入れ子の交差検証：学習用2941窓を記録単位で5分割し、4分割だけで後ろ向き選択をやり直す。
%     選んだ列で、選択に使わなかった1分割を予測する。選び方まで含めた正解率になる。
%     内側の選択は、4分割の中の記録単位3分割（拡張なし）の正解率で、32列から1列まで外していく。
%     外側の評価は、本番と同じく拡張8通りを足して学習する。
%   2 検証用407窓・テスト201窓：学習用2941窓全体で同じ選択をし、学習用だけで学習して予測する。
%     この2つは選択に一度も使っていない。
%   3 無作為の列：同じ列数を32列から無作為に20通り選び、検証用407窓の正解数を比べる。
%     選んだ組が無作為より上なら、列の選び方が効いている。
%   比較のため、主催者の12列（6軸の平均と標準偏差）も同じ物差しで測る。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
ver = har.readVerification(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
Yv = categorical(ver.labels, data.classes);
g = recordingGroups(cat(1, raw, data.testRaw));
g = g(1:numel(Y));
augFns = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
X = productionFeatures(raw);
Xaug = cellfun(@(f) productionFeatures(f(raw)), augFns, 'UniformOutput', false);
Xte = productionFeatures(data.testRaw);
Xv = productionFeatures(ver.raw);
sizes = [32 20 12 8 6 4 3 2 1];

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

ids = unique(g);
[~, pos] = ismember(g, ids);
s0 = RandStream('twister', 'Seed', 20261116);
fold = mod(randperm(s0, numel(ids)), 5)' + 1;
fold = fold(pos);

% 1 入れ子の交差検証
pred = repmat(Y, 1, numel(sizes));
chosen = cell(5, numel(sizes));
for k = 1:5
    tr = find(fold ~= k);
    te = fold == k;
    sets = backward(X(tr, :), Y(tr), g(tr), sizes, 31 + k);
    for j = 1:numel(sizes)
        c = sets{j};
        chosen{k, j} = c;
        f = fitAug(X, Xaug, Y, tr, c);
        pred(te, j) = f(X(te, c));
    end
    fprintf('外側の分割 %d：12列 %s\n', k, strjoin(string(sets{sizes == 12}), ' '));
end
% 2 学習用全体で選び、検証用・テストで測る
setsAll = backward(X, Y, g, sizes, 41);
rows = {};
for j = 1:numel(sizes)
    c = setsAll{j};
    f = fitAug(X, Xaug, Y, 1:numel(Y), c);
    pv = f(Xv(:, c)); pt = f(Xte(:, c));
    nestedAcc = mean(pred(:, j) == Y);
    ssNested = sum(ismember(Y, categorical(["Sitting", "Standing"])) & ismember(pred(:, j), categorical(["Sitting", "Standing"])) & pred(:, j) ~= Y);
    ssVer = sum(ismember(Yv, categorical(["Sitting", "Standing"])) & ismember(pv, categorical(["Sitting", "Standing"])) & pv ~= Yv);
    % 外側の5分割で選ばれた列が、全体で選んだ列とどれだけ重なるか
    overlap = mean(cellfun(@(q) numel(intersect(q, c)) / numel(c), chosen(:, j)));
    rows(end + 1, :) = {sizes(j), nestedAcc, ssNested, sum(pv == Yv), ssVer, sum(pt == Yte), overlap, strjoin(string(c), ' ')}; %#ok<AGROW>
    fprintf('%2d列  入れ子 %.4f (座位↔立位 %d)  検証用 %d/407 (座位↔立位 %d)  テスト %d/201  列の重なり %.2f  [%s]\n', ...
        sizes(j), nestedAcc, ssNested, sum(pv == Yv), ssVer, sum(pt == Yte), overlap, strjoin(string(c), ' '));
end
T = cell2table(rows, 'VariableNames', {'nFeatures', 'nestedCv', 'nestedSitStand', 'verCorrect', 'verSitStand', ...
    'testCorrect', 'foldOverlap', 'columns'});
writetable(T, fullfile(out, 'stage19_nested.csv'), 'Encoding', 'UTF-8');

% 3 無作為の列
rs = RandStream('twister', 'Seed', 20261126);
randRows = {};
for n = [20 12 8 6 4 2]
    draws = arrayfun(@(~) sort(randperm(rs, 32, n)), 1:20, 'UniformOutput', false);
    v = zeros(20, 1);
    parfor d = 1:20
        c = draws{d};
        f = fitAug(X, Xaug, Y, 1:numel(Y), c);
        v(d) = sum(f(Xv(:, c)) == Yv);
    end
    randRows(end + 1, :) = {n, mean(v), median(v), min(v), max(v)}; %#ok<AGROW>
    fprintf('無作為 %2d列  検証用 平均 %.1f  中央値 %.1f  最小 %d  最大 %d\n', n, mean(v), median(v), min(v), max(v));
end
T3 = cell2table(randRows, 'VariableNames', {'nFeatures', 'verMean', 'verMedian', 'verMin', 'verMax'});
writetable(T3, fullfile(out, 'stage19_random.csv'), 'Encoding', 'UTF-8');

% 主催者の12列
O = organizer12(raw);
Oaug = cellfun(@(f) organizer12(f(raw)), augFns, 'UniformOutput', false);
po = Y;
for k = 1:5
    tr = find(fold ~= k);
    f = fitAug(O, Oaug, Y, tr, 1:12);
    po(fold == k) = f(O(fold == k, :));
end
f = fitAug(O, Oaug, Y, 1:numel(Y), 1:12);
fprintf('主催者の12列  交差検証 %.4f  検証用 %d/407  テスト %d/201\n', mean(po == Y), ...
    sum(f(organizer12(ver.raw)) == Yv), sum(f(organizer12(data.testRaw)) == Yte));
fprintf('STAGE19_DONE\n');
end


function sets = backward(X, Y, g, sizes, seed)
% 32列から、外しても内側の正解率が最も下がらない列を1つずつ外す。
ids = unique(g);
[~, pos] = ismember(g, ids);
s = RandStream('twister', 'Seed', seed);
fold = mod(randperm(s, numel(ids)), 3)' + 1;
fold = fold(pos);
current = 1:size(X, 2);
sets = cell(1, numel(sizes));
sets{sizes == numel(current)} = current;
while numel(current) > min(sizes)
    cand = arrayfun(@(c) setdiff(current, c), current, 'UniformOutput', false);
    a = zeros(numel(cand), 1);
    parfor j = 1:numel(cand)
        a(j) = innerAcc(X(:, cand{j}), Y, fold);
    end
    [~, k] = max(a);
    current = cand{k};
    if any(sizes == numel(current))
        sets{sizes == numel(current)} = current;
    end
end
end


function a = innerAcc(X, Y, fold)
pred = Y;
opts = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(size(X, 2)));
for k = 1:3
    tr = fold ~= k;
    f = fitModel('gauss_svm', X(tr, :), Y(tr), opts);
    pred(~tr) = f(X(~tr, :));
end
a = mean(pred == Y);
end


function f = fitAug(X, Xaug, Y, tr, c)
Xtr = [X(tr, c); cell2mat(cellfun(@(A) A(tr, c), Xaug(:), 'UniformOutput', false))];
f = fitModel('gauss_svm', Xtr, repmat(Y(tr), numel(Xaug) + 1, 1), ...
    struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(numel(c))));
end


function X = organizer12(raw)
m = squeeze(mean(raw, 2));
s = squeeze(std(raw, 0, 2));
X = [m(:, 1:3), s(:, 1:3), m(:, 4:6), s(:, 4:6)];
end
