function [acc, predicted, perRepeat] = groupedCV(X, Y, groups, method, repeats, opts, augmentFcn)
% 記録単位の5分割交差検証を repeats 回くり返す。
%
%   X, Y       : 特徴量と正解（categorical）
%   groups     : 記録番号。同じ番号の窓は同じ分割に入る
%   augmentFcn : 省略可。学習側の行番号を受け取り [Xaug, Yaug] を返す関数。
%                拡張した窓は学習側だけに入れ、評価側には入れない
%   acc        : 全くり返しの平均正解率
%   predicted  : 1回目のくり返しの予測（混同行列用）
%   perRepeat  : くり返しごとの正解率（ばらつきの確認用）
if nargin < 5 || isempty(repeats), repeats = 1; end
if nargin < 6 || isempty(opts), opts = struct(); end
if nargin < 7, augmentFcn = []; end

ids = unique(groups);
[~, position] = ismember(groups, ids);
perRepeat = zeros(repeats, 1);
predicted = Y;
for r = 1:repeats
    stream = RandStream('twister', 'Seed', 20261115 + r);
    foldOfGroup = mod(randperm(stream, numel(ids)), 5)' + 1;
    fold = foldOfGroup(position);
    pred = Y;
    for k = 1:5
        held = fold == k;
        trainRows = find(~held);
        if isempty(augmentFcn)
            Xtr = X(trainRows, :);
            Ytr = Y(trainRows);
        else
            [Xtr, Ytr] = augmentFcn(trainRows);
        end
        rng(r * 100 + k, 'twister');
        f = fitModel(method, Xtr, Ytr, opts);
        pred(held) = f(X(held, :));
    end
    perRepeat(r) = mean(pred == Y);
    if r == 1, predicted = pred; end
end
acc = mean(perRepeat);
end
