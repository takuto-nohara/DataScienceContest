% 公表している分類結果が偽装・曲解されていないかを、学習し直して確かめる。
%
% 確認する内容:
%   1. 保存済みモデルの予測が、リポジトリに記録した予測CSVと一致するか
%   2. 正解率を単純な一致数から数え直しても同じ値になるか
%   3. spec.json だけから学習し直したモデルが、保存済みモデルと同じ予測をするか
%      (保存済みモデルにテストデータが混入していれば、ここで食い違う)
%   4. テスト窓と同一、またはほぼ同一の窓が学習データに含まれていないか
%   5. テストラベルを見ずに決めた場合の正解率 (学習データ内の交差検証)
%
% このスクリプトはモデルファイルを書き換えない。結果は artifacts/ に出力する。

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

data = har.loadData(root);
protocol = har.protocol(root, data);
validationRaw = data.trainRaw(protocol.validationRows, :, :);
validationLabels = data.trainLabels(protocol.validationRows);

summary = struct();

%% 1・2. 保存済みモデルの予測と、記録済みCSVおよび正解数を照合する
modelIds = "proto_v" + (1:10);
recorded = readtable(fullfile(root, 'evaluation', 'refinement_comparison', ...
    'leaderboard.csv'), 'TextType', 'string');
replay = struct([]);

for k = 1:numel(modelIds)
    id = modelIds(k);
    saved = load(fullfile(root, 'model', id, 'trained.mat'), 'bundle');
    bundle = saved.bundle;

    testPrediction = har.predict(bundle, data.testRaw);
    validationPrediction = har.predict(bundle, validationRaw);

    csvFile = fullfile(root, 'evaluation', 'refinement_comparison', ...
        id + "_predictions.csv");
    csv = readtable(csvFile, 'TextType', 'string');
    assert(isequal(csv.truth, data.testLabels), ...
        '%s: 記録された正解ラベルが配布データと異なる。', id);
    assert(isequal(csv.prediction, testPrediction), ...
        '%s: 記録された予測が再実行の結果と異なる。', id);

    testCorrect = sum(testPrediction == data.testLabels);
    validationCorrect = sum(validationPrediction == validationLabels);

    row = recorded(recorded.model == id, :);
    assert(row.testCorrect == testCorrect, ...
        '%s: 記録された正解数が再計算と異なる。', id);
    assert(abs(row.validationAccuracy - validationCorrect / 735) < 1e-12, ...
        '%s: 記録された検証正解率が再計算と異なる。', id);

    replay(k).model = id;
    replay(k).testCorrect = testCorrect;
    replay(k).validationCorrect = validationCorrect;
    fprintf('%-9s 再実行: テスト %3d/201, 検証 %3d/735 (記録と一致)\n', ...
        id, testCorrect, validationCorrect);
end
summary.replay = replay;

%% 3. spec.json だけから学習し直し、保存済みモデルと予測を比べる
fitRaw = data.trainRaw(protocol.fitRows, :, :);
fitLabels = data.trainLabels(protocol.fitRows);
retrained = struct([]);

for id = ["proto_v8", "proto_v9", "proto_v10"]
    saved = load(fullfile(root, 'model', id, 'trained.mat'), 'bundle');
    bundle = saved.bundle;
    spec = bundle.spec;

    trainRaw = fitRaw;
    trainLabels = fitLabels;
    if isfield(spec, 'augmentationGains')
        [trainRaw, trainLabels] = har.augment( ...
            trainRaw, trainLabels, spec.augmentationGains);
    end

    features = har.features(trainRaw, spec.featureSet);
    center = mean(features, 1);
    scale = std(features, 0, 1);
    scale(scale < 1e-12) = 1;
    assert(max(abs(center - bundle.center)) < 1e-10 ...
        && max(abs(scale - bundle.scale)) < 1e-10, ...
        '%s: 保存された標準化の値が学習用の行から再現できない。', id);

    rng(spec.seed, 'twister');
    learner = templateSVM('KernelFunction', 'gaussian', ...
        'KernelScale', spec.kernelScale, ...
        'BoxConstraint', spec.boxConstraint, 'Standardize', false);
    primary = fitcecoc((features - center) ./ scale, trainLabels, ...
        'Learners', learner, 'Coding', 'onevsone');

    % 保存済みモデルの主SVMだけを差し替え、残りの処理は同じものを通す。
    rebuilt = bundle;
    if strcmp(spec.algorithm, 'static_vote_svm')
        rebuilt.estimator.primary = compact(primary);
    else
        rebuilt.estimator = compact(primary);
    end

    savedTest = har.predict(bundle, data.testRaw);
    rebuiltTest = har.predict(rebuilt, data.testRaw);
    savedValidation = har.predict(bundle, validationRaw);
    rebuiltValidation = har.predict(rebuilt, validationRaw);

    entry = struct('model', id, ...
        'testDisagreements', sum(savedTest ~= rebuiltTest), ...
        'validationDisagreements', sum(savedValidation ~= rebuiltValidation), ...
        'retrainedTestCorrect', sum(rebuiltTest == data.testLabels));
    if isempty(retrained), retrained = entry; else, retrained(end + 1) = entry; end %#ok<SAGROW>
    fprintf('%-9s 学習し直し: テスト %d/201, 保存モデルとの不一致 %d件 (検証 %d件)\n', ...
        id, entry.retrainedTestCorrect, entry.testDisagreements, ...
        entry.validationDisagreements);
end
summary.retrained = retrained;

%% 4. テスト窓と同一・ほぼ同一の窓が学習データにないか
flatTrain = reshape(data.trainRaw, size(data.trainRaw, 1), []);
flatTest = reshape(data.testRaw, size(data.testRaw, 1), []);
[nearestDistance, nearestRow] = min(pdist2(flatTest, flatTrain), [], 2);

% 窓が50%重なる記録なら、前半64点と後半64点が一致するはずである。
firstHalf = reshape(data.testRaw(:, 1:64, :), size(data.testRaw, 1), []);
secondHalf = reshape(data.testRaw(:, 65:128, :), size(data.testRaw, 1), []);
trainFirstHalf = reshape(data.trainRaw(:, 1:64, :), size(data.trainRaw, 1), []);
trainSecondHalf = reshape(data.trainRaw(:, 65:128, :), size(data.trainRaw, 1), []);
overlapsNext = min(pdist2(secondHalf, trainFirstHalf), [], 2) < 1e-9;
overlapsPrevious = min(pdist2(firstHalf, trainSecondHalf), [], 2) < 1e-9;

summary.duplicates = struct( ...
    'exactDuplicates', sum(nearestDistance < 1e-9), ...
    'minimumDistance', min(nearestDistance), ...
    'medianDistance', median(nearestDistance), ...
    'testWindowsSharingHalfWithTraining', sum(overlapsNext | overlapsPrevious), ...
    'nearestNeighborSameLabel', ...
        mean(data.trainLabels(nearestRow) == data.testLabels));
fprintf('完全一致の窓: %d件 / 学習窓と64点を共有するテスト窓: %d件\n', ...
    summary.duplicates.exactDuplicates, ...
    summary.duplicates.testWindowsSharingHalfWithTraining);

%% 5. テストラベルを使わない見積もり: 学習データ2941件の5分割交差検証
rng(20261111, 'twister');
folds = cvpartition(categorical(data.trainLabels), 'KFold', 5);
crossValidation = struct([]);

for id = ["proto_v3", "proto_v8", "proto_v9", "proto_v10"]
    saved = load(fullfile(root, 'model', id, 'trained.mat'), 'bundle');
    spec = saved.bundle.spec;
    correct = 0;

    for fold = 1:folds.NumTestSets
        trainRows = find(training(folds, fold));
        heldOutRows = find(test(folds, fold));
        prediction = fitAndPredict(spec, data.trainRaw(trainRows, :, :), ...
            data.trainLabels(trainRows), data.trainRaw(heldOutRows, :, :));
        correct = correct + sum(prediction == data.trainLabels(heldOutRows));
    end

    entry = struct('model', id, 'correct', correct, ...
        'total', numel(data.trainLabels), ...
        'accuracy', correct / numel(data.trainLabels));
    if isempty(crossValidation), crossValidation = entry; else, crossValidation(end + 1) = entry; end %#ok<SAGROW>
    fprintf('%-9s 5分割交差検証: %d/%d = %.2f%%\n', ...
        id, correct, entry.total, 100 * entry.accuracy);
end
summary.crossValidation = crossValidation;

%% 6. 重なり合う窓を同じ組にまとめた交差検証
% 64点を共有する窓どうしをつなぎ、連続した記録を1つの組とみなす。
% 同じ組の窓は学習側と評価側に分かれないので、窓の重なりによる水増しを除ける。
allRaw = cat(1, data.trainRaw, data.testRaw);
windowCount = size(allRaw, 1);
headHalf = reshape(allRaw(:, 1:64, :), windowCount, []);
tailHalf = reshape(allRaw(:, 65:128, :), windowCount, []);
[follower, leader] = find(pdist2(headHalf, tailHalf) < 1e-9);
recording = conncomp(graph(leader, follower, [], windowCount))';
trainRecording = recording(1:numel(data.trainLabels));

recordingIds = unique(trainRecording);
rng(20261111, 'twister');
foldOfRecording = mod(randperm(numel(recordingIds)), 5)' + 1;
[~, position] = ismember(trainRecording, recordingIds);
foldOfWindow = foldOfRecording(position);

groupedValidation = struct([]);
for id = ["proto_v3", "proto_v8", "proto_v9", "proto_v10"]
    saved = load(fullfile(root, 'model', id, 'trained.mat'), 'bundle');
    spec = saved.bundle.spec;
    correct = 0;

    for fold = 1:5
        heldOut = foldOfWindow == fold;
        prediction = fitAndPredict(spec, data.trainRaw(~heldOut, :, :), ...
            data.trainLabels(~heldOut), data.trainRaw(heldOut, :, :));
        correct = correct + sum(prediction == data.trainLabels(heldOut));
    end

    entry = struct('model', id, 'correct', correct, ...
        'total', numel(data.trainLabels), ...
        'accuracy', correct / numel(data.trainLabels));
    if isempty(groupedValidation), groupedValidation = entry; else, groupedValidation(end + 1) = entry; end %#ok<SAGROW>
    fprintf('%-9s 記録単位の5分割交差検証: %d/%d = %.2f%%\n', ...
        id, correct, entry.total, 100 * entry.accuracy);
end
summary.groupedCrossValidation = groupedValidation;
summary.recordings = struct('count', numel(recordingIds), ...
    'medianWindows', median(groupcounts(trainRecording)), ...
    'testWindowsInsideTrainingRecordings', ...
        sum(ismember(recording(numel(data.trainLabels) + 1:end), recordingIds)));
fprintf('連続した記録の数: %d / 学習側の記録に挟まるテスト窓: %d件\n', ...
    summary.recordings.count, ...
    summary.recordings.testWindowsInsideTrainingRecordings);

%% 探索回数: 201件のテストラベルを何通りの設定で採点したか
searchFiles = ["search.csv", "hierarchy_search.csv", "augmentation_search.csv"];
attempts = 0;
for file = searchFiles
    attempts = attempts + height(readtable( ...
        fullfile(root, 'evaluation', 'refinement', file)));
end
summary.testAwareSearchAttempts = attempts;
fprintf('テストラベルで採点した設定の数: %d通り\n', attempts);

har.writeJson(fullfile(root, 'artifacts', 'integrity-check.json'), summary);
fprintf('INTEGRITY_CHECK_DONE\n');


function prediction = fitAndPredict(spec, trainRaw, trainLabels, heldOutRaw)
% spec の設定で学習し、未使用の窓を予測する。proto_v10 は補助モデルも作り直す。
plainFeatures = har.features(trainRaw, 'stats12');
plainHeldOut = har.features(heldOutRaw, 'stats12');
plainLabels = trainLabels;

if isfield(spec, 'augmentationGains')
    [trainRaw, trainLabels] = har.augment( ...
        trainRaw, trainLabels, spec.augmentationGains);
end

features = har.features(trainRaw, spec.featureSet);
heldOut = har.features(heldOutRaw, spec.featureSet);
[z, zHeldOut] = standardize(features, heldOut);

width = 'auto';
box = 1;
if isfield(spec, 'kernelScale'), width = spec.kernelScale; end
if isfield(spec, 'boxConstraint'), box = spec.boxConstraint; end

rng(spec.seed, 'twister');
learner = templateSVM('KernelFunction', 'gaussian', ...
    'KernelScale', width, 'BoxConstraint', box, 'Standardize', false);
primary = fitcecoc(z, trainLabels, 'Learners', learner, 'Coding', 'onevsone');
prediction = string(predict(primary, zHeldOut));

if ~strcmp(spec.algorithm, 'static_vote_svm'), return; end

[zPlain, zPlainHeldOut] = standardize(plainFeatures, plainHeldOut);
tree = fitctree(zPlain, plainLabels, 'MaxNumSplits', 100, 'MinLeafSize', 5);
linearLearner = templateSVM('KernelFunction', 'linear', ...
    'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', false);
linear = fitcecoc(zPlain, plainLabels, ...
    'Learners', linearLearner, 'Coding', 'onevsone');

votes = [prediction, string(predict(tree, zPlainHeldOut)), ...
    string(predict(linear, zPlainHeldOut))];
prediction = har.staticVote(prediction, votes);
end


function [z, zOther] = standardize(features, other)
% 学習側の平均と標準偏差だけで両方を標準化する。
center = mean(features, 1);
scale = std(features, 0, 1);
scale(scale < 1e-12) = 1;
z = (features - center) ./ scale;
zOther = (other - center) ./ scale;
end
