function results = search_single_model(root)
% 多数決を使わない1つのモデルで、開発用201件をどこまで当てられるかを探す。
%
%   root    : リポジトリのフォルダ (省略時はこのファイルの1つ上)
%   results : 試した設定ごとの結果の表
%
% 分類学習器 (Classification Learner) で再現できる形だけを試す。
%   - 学習には学習用2941件をすべて使う (分類学習器と同じ)
%   - データ拡張はしない
%   - 標準化は分類器の 'Standardize' に任せる (Pretreatment では行わない)
%   - 多クラスSVMは分類学習器の既定と同じ one-vs-one
%
% 開発用201件の正解を見て設定を選ぶことになるので、試した設定の数を必ず
% 記録し、報告する。各設定について5分割交差検証の正解率も記録する。

if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
end
addpath(fullfile(root, 'matlab'));

data = har.loadData(root);
outputFolder = fullfile(root, 'evaluation', 'single_model_search');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

featureSets = ["spectral32", "gravity32", "shape32", "window32", ...
    "distribution32", "temporal32"];
candidates = buildCandidates();

rows = {};
for featureSet = featureSets
    trainX = har.features(data.trainRaw, char(featureSet));
    testX = har.features(data.testRaw, char(featureSet));
    trainY = categorical(data.trainLabels);
    testY = categorical(data.testLabels);

    for k = 1:numel(candidates)
        candidate = candidates(k);
        rng(1, 'twister');
        model = candidate.fit(trainX, trainY);
        predicted = predict(model, testX);
        correct = sum(predicted == testY);
        fprintf('%-15s %-28s test %3d/%d\n', featureSet, candidate.name, ...
            correct, numel(testY));
        rows(end + 1, :) = {featureSet, string(candidate.name), correct, ...
            NaN, find(predicted ~= testY)'}; %#ok<AGROW>
    end
end

results = cell2table(rows, 'VariableNames', ...
    {'featureSet', 'model', 'testCorrect', 'cvAccuracy', 'missedRows'});

% 交差検証は時間がかかるので、開発用で199件以上の設定だけに行う。
for k = find(results.testCorrect >= 199)'
    trainX = har.features(data.trainRaw, char(results.featureSet(k)));
    trainY = categorical(data.trainLabels);
    candidate = candidates(strcmp({candidates.name}, results.model(k)));
    results.cvAccuracy(k) = crossValidate(candidate, trainX, trainY);
    fprintf('CV %-15s %-28s %.4f\n', results.featureSet(k), ...
        results.model(k), results.cvAccuracy(k));
end

results.missedRows = cellfun(@(r) strjoin(string(r), ' '), ...
    results.missedRows, 'UniformOutput', false);
writetable(results, fullfile(outputFolder, 'results.csv'));
fprintf('Tried %d settings.\n', height(results));
end


function accuracy = crossValidate(candidate, x, y)
% 分類学習器の既定と同じ5分割交差検証の正解率を返す。
rng(1, 'twister');
partition = cvpartition(y, 'KFold', 5);
correct = 0;
for fold = 1:partition.NumTestSets
    model = candidate.fit(x(training(partition, fold), :), ...
        y(training(partition, fold)));
    heldOut = test(partition, fold);
    correct = correct + sum(predict(model, x(heldOut, :)) == y(heldOut));
end
accuracy = correct / numel(y);
end


function candidates = buildCandidates()
% 分類学習器で選べるモデルと、その主なハイパーパラメータの組み合わせ。
candidates = struct('name', {}, 'fit', {});

scales = [1, 1.41, 2, 2.83, 4, 5.66, 8, 11.3];
boxes = [0.5, 1, 2, 5, 10, 20, 50, 100];
for s = scales
    for c = boxes
        learner = templateSVM('KernelFunction', 'gaussian', ...
            'KernelScale', s, 'BoxConstraint', c, 'Standardize', true);
        candidates(end + 1) = makeCandidate( ...
            sprintf('svm_gauss_s%.2f_c%g', s, c), learner); %#ok<AGROW>
    end
end

for order = [2, 3]
    for c = [0.1, 1, 10]
        learner = templateSVM('KernelFunction', 'polynomial', ...
            'PolynomialOrder', order, 'BoxConstraint', c, 'Standardize', true);
        candidates(end + 1) = makeCandidate( ...
            sprintf('svm_poly%d_c%g', order, c), learner); %#ok<AGROW>
    end
end

for neighbors = [1, 5, 10]
    candidates(end + 1) = struct('name', sprintf('knn_%d', neighbors), ...
        'fit', @(x, y) fitcknn(x, y, 'NumNeighbors', neighbors, ...
        'Standardize', true, 'DistanceWeight', 'squaredinverse')); %#ok<AGROW>
end

for layer = [10, 25, 50, 100]
    for lambda = [0, 1e-4, 1e-3]
        candidates(end + 1) = struct( ...
            'name', sprintf('net_%d_l%g', layer, lambda), ...
            'fit', @(x, y) fitcnet(x, y, 'LayerSizes', layer, ...
            'Lambda', lambda, 'Standardize', true)); %#ok<AGROW>
    end
end

candidates(end + 1) = struct('name', 'bagged_trees', ...
    'fit', @(x, y) fitcensemble(x, y, 'Method', 'Bag', ...
    'NumLearningCycles', 100));
candidates(end + 1) = struct('name', 'subspace_discriminant', ...
    'fit', @(x, y) fitcensemble(x, y, 'Method', 'Subspace', ...
    'Learners', 'discriminant', 'NumLearningCycles', 30));
end


function candidate = makeCandidate(name, learner)
candidate = struct('name', name, 'fit', @(x, y) fitcecoc(x, y, ...
    'Learners', learner, 'Coding', 'onevsone'));
end
