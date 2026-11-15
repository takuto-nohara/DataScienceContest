function stage2d_knn_tune(root)
% 第2d段：k-NN の前向き選択で得た列について、近傍数・重み・距離を比べる。
% 分類学習器の k-NN で設定できる項目だけを動かす。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
load(fullfile(out, 'pool.mat'), 'Xtr', 'Xte', 'Ytr', 'Yte', 'names', 'gtr');
F = readtable(fullfile(out, 'stage2_forward_knn.csv'), 'TextType', 'string');

rows = {};
for nf = [9, 12, 14]
    cols = arrayfun(@(s) find(names == s), F.addedFeature(1:nf))';
    X = Xtr(:, cols);
    mu = mean(X); sd = std(X);
    Z = (X - mu) ./ sd;
    Zte = (Xte(:, cols) - mu) ./ sd;
    for k = [1, 3, 5, 7, 10, 15, 20, 30]
        for w = ["equal", "inverse", "squaredinverse"]
            for d = ["euclidean", "cityblock"]
                acc = zeros(5, 1);
                ids = unique(gtr); [~, pos] = ismember(gtr, ids);
                for r = 1:5
                    s = RandStream('twister', 'Seed', 20261115 + r);
                    fold = mod(randperm(s, numel(ids)), 5)' + 1; fold = fold(pos);
                    pred = Ytr;
                    for f = 1:5
                        h = fold == f;
                        m = fitcknn(X(~h, :), Ytr(~h), 'NumNeighbors', k, ...
                            'DistanceWeight', w, 'Distance', d, 'Standardize', true);
                        pred(h) = predict(m, X(h, :));
                    end
                    acc(r) = mean(pred == Ytr);
                end
                m = fitcknn(Z, Ytr, 'NumNeighbors', k, 'DistanceWeight', w, 'Distance', d);
                t = mean(predict(m, Zte) == Yte);
                rows(end + 1, :) = {nf, k, w, d, mean(acc), std(acc), min(acc), t}; %#ok<AGROW>
                fprintf('nf %2d k %2d %-15s %-9s CV %.4f sd %.4f test %.4f\n', nf, k, w, d, mean(acc), std(acc), t);
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'nFeatures', 'k', 'weight', 'distance', ...
    'cvMean', 'cvSd', 'cvWorst', 'test201'});
writetable(T, fullfile(out, 'stage2d_knn_tuning.csv'));
fprintf('STAGE2D_DONE\n');
end
