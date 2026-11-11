function result = metrics(truth, predicted, classes)
% 正解と予測から、正解率・混同行列・クラスごとの指標を求める。
%
%   truth, predicted : 同じ長さの行動名
%   classes          : 行動名の一覧。混同行列の行と列はこの順に並ぶ
%
% 混同行列は、行が正解、列が予測を表す。

truth = string(truth(:));
predicted = string(predicted(:));
classes = string(classes(:));
assert(numel(truth) == numel(predicted) && ~isempty(truth), ...
    'Invalid prediction count.');
assert(all(ismember(truth, classes)) && all(ismember(predicted, classes)), ...
    'Unknown label.');

classCount = numel(classes);
confusion = zeros(classCount);
for actual = 1:classCount
    for guess = 1:classCount
        confusion(actual, guess) = sum( ...
            truth == classes(actual) & predicted == classes(guess));
    end
end

truePositives = diag(confusion);
support = sum(confusion, 2);
predictedCount = sum(confusion, 1)';

% 件数が0のクラスで0除算にならないよう、分母を1以上にしておく。
precision = truePositives ./ max(predictedCount, 1);
recall = truePositives ./ max(support, 1);
f1 = 2 * precision .* recall ./ max(precision + recall, eps);

total = sum(confusion(:));
result = struct( ...
    'accuracy', sum(truePositives) / total, ...
    'macroF1', mean(f1), ...
    'balancedAccuracy', mean(recall(support > 0)), ...
    'correct', sum(truePositives), ...
    'total', total, ...
    'classes', classes, ...
    'confusion', confusion, ...
    'precision', precision, ...
    'recall', recall, ...
    'f1', f1, ...
    'support', support);
end
