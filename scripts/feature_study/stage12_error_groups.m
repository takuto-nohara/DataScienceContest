function stage12_error_groups(root)
% 第12段：残る座位↔立位の誤りが、記録（64点を共有してつながる窓のまとまり）ごとに
% 固まっているかを調べる。固まっていれば、人や装着の違いによる誤りで、1窓の特徴量を
% 足しても直しにくい。ばらばらなら、窓ごとの揺らぎによる誤りである。
% 本番の32列（productionFeatures）、拡張8通り、ガウスSVM（KS 7.9、C 10）、記録単位の
% 交差検証5回。テスト201窓は使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
g = groups(1:numel(Y));
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
    perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
X = productionFeatures(raw);
Xaug = cellfun(@productionFeatures, aug, 'UniformOutput', false);
opts = struct('BoxConstraint', 10, 'KernelScale', 7.9);
ids = unique(g);
[~, pos] = ismember(g, ids);
repeats = 5;
wrong = zeros(numel(Y), 1);
for rep = 1:repeats
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    parts = cell(5, 1);
    parfor k = 1:5
        tr = fold ~= k;
        Xtr = [X(tr, :); cell2mat(cellfun(@(A) A(tr, :), Xaug(:), 'UniformOutput', false))];
        Ytr = repmat(Y(tr), numel(Xaug) + 1, 1);
        f = fitModel('gauss_svm', Xtr, Ytr, opts);
        parts{k} = f(X(~tr, :));
    end
    for k = 1:5
        pred(fold == k) = parts{k};
    end
    wrong = wrong + (pred ~= Y & ismember(Y, categorical(["Sitting", "Standing"])) & ...
        ismember(pred, categorical(["Sitting", "Standing"])));
    fprintf('repeat %d done\n', rep);
end

static = ismember(Y, categorical(["Sitting", "Standing"]));
gid = unique(g(static));
rows = zeros(numel(gid), 4);
for i = 1:numel(gid)
    in = g == gid(i) & static;
    rows(i, :) = [gid(i), sum(in), sum(wrong(in)) / repeats, mean(wrong(in) > 0)];
end
T = array2table(rows, 'VariableNames', {'recording', 'windows', 'meanErrors', 'fractionEverWrong'});
T.label = string(arrayfun(@(r) string(mode(Y(g == r & static))), T.recording));
T = sortrows(T, 'meanErrors', 'descend');
writetable(T, fullfile(out, 'stage12_error_groups.csv'));
total = sum(T.meanErrors);
withErr = T(T.meanErrors > 0, :);
cum = cumsum(withErr.meanErrors) / total;
fprintf('sit/stand recordings %d (windows %d), mean errors %.1f\n', height(T), sum(T.windows), total);
fprintf('recordings with any error %d; top 5 hold %.0f%%, top 10 hold %.0f%% of errors\n', ...
    height(withErr), 100 * cum(min(5, end)), 100 * cum(min(10, end)));
fprintf('recordings where >=50%% of windows are wrong on average: %d (errors %.1f)\n', ...
    sum(T.meanErrors ./ T.windows >= 0.5), sum(T.meanErrors(T.meanErrors ./ T.windows >= 0.5)));
disp(T(1:min(15, height(T)), :));
fprintf('STAGE12_DONE\n');
end
