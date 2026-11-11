% 保存済みモデル proto_v10 が「実際に何を根拠に判定したか」を reports/ に書き出す。
% モデルの学習は一切行わず、保存済みのモデルで予測をやり直すだけである。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% データと保存済みモデルを読み込む
modelId = 'proto_v10';
modelFile = fullfile(root, 'model/proto_v10/trained.mat');
data = har.loadData(root);
saved = load(modelFile, 'bundle');
bundle = saved.bundle;
primaryModel = bundle.estimator.primary;

outputFolder = fullfile(root, 'reports');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% テスト窓ごとに、各モデルの予測と最終判定を記録する
% 主 SVM の予測と、クラスごとの負の損失 (大きいほどそのクラスらしい) を求める
[testFeatures, featureNames] = har.features(data.testRaw, bundle.spec.featureSet);
[primaryPrediction, negativeLoss] = predict(primaryModel, ...
    standardize(testFeatures, bundle.center, bundle.scale));
primaryPrediction = string(primaryPrediction);

% 静止クラスの見直しに加わる2つの補助モデルと、多数決を経た最終判定
treeSpecialist = bundle.estimator.specialists{1};
linearSpecialist = bundle.estimator.specialists{2};
treePrediction = har.predict(treeSpecialist, data.testRaw);
linearPrediction = har.predict(linearSpecialist, data.testRaw);
finalPrediction = har.predict(bundle, data.testRaw);

% 1位と2位の差が小さいほど、主 SVM が迷った窓である
sortedLoss = sort(negativeLoss, 2, 'descend');
topTwoGap = sortedLoss(:, 1) - sortedLoss(:, 2);

% 多数決は、主 SVM が静止クラスと予測した窓にだけ適用される
staticClassNames = {'Sitting', 'Standing', 'Laying'};
voteApplied = ismember(primaryPrediction, staticClassNames);
voteChangedPrediction = primaryPrediction ~= finalPrediction;

testWindowCount = numel(primaryPrediction);
trace = table((1:testWindowCount)', data.testLabels, primaryPrediction, ...
    treePrediction, linearPrediction, finalPrediction, ...
    voteApplied, voteChangedPrediction, topTwoGap, ...
    'VariableNames', {'row', 'truth', 'primarySVM', 'tree', 'linearSVM', 'final', ...
    'voteApplied', 'voteChangedPrediction', 'primaryTopTwoNegLossGap'});
writetable(trace, fullfile(outputFolder, 'classification-trace.csv'), ...
    'Encoding', 'UTF-8');

%% 決定木の分岐条件を、標準化前の単位に戻して記録する
treeModel = treeSpecialist.estimator;
branchNodes = find(treeModel.IsBranchNode);
cuts = struct([]);

for k = 1:numel(branchNodes)
    node = branchNodes(k);
    featureIndex = find( ...
        string(treeModel.PredictorNames) == string(treeModel.CutPredictor{node}));

    % 木は標準化後の値で分岐するので、元の単位のしきい値も併記する
    standardizedCut = treeModel.CutPoint(node);
    rawCut = standardizedCut * treeSpecialist.scale(featureIndex) ...
        + treeSpecialist.center(featureIndex);

    record = struct('node', node, ...
        'feature', treeSpecialist.featureNames(featureIndex), ...
        'standardizedCut', standardizedCut, ...
        'rawCut', rawCut, ...
        'leftChild', treeModel.Children(node, 1), ...
        'rightChild', treeModel.Children(node, 2));
    cuts = appendRecord(cuts, record);
end

writetable(struct2table(cuts), fullfile(outputFolder, 'tree-branch-criteria.csv'), ...
    'Encoding', 'UTF-8');

%% 発表で取り上げる窓の判定内容をまとめる
% 改良前のモデルが誤った5件 (8, 10, 105, 176, 201) と、多数決で予測が変わる155件目
focusRows = [8; 10; 105; 155; 176; 201];
focusRecords = struct([]);

for k = 1:numel(focusRows)
    row = focusRows(k);
    record = struct('row', row, ...
        'truth', data.testLabels(row), ...
        'primary', primaryPrediction(row), ...
        'tree', treePrediction(row), ...
        'linearSVM', linearPrediction(row), ...
        'final', finalPrediction(row), ...
        'primaryNegativeLoss', negativeLoss(row, :));
    focusRecords = appendRecord(focusRecords, record);
end

disclaimer = ['Labels guided prior development. Negative-loss values and gaps ', ...
    'are not calibrated probabilities or causal feature attributions.'];
evidence = struct('model', modelId, ...
    'modelSHA256', har.hashFile(modelFile), ...
    'featureNames', featureNames, ...
    'primaryCenter', bundle.center, ...
    'primaryScale', bundle.scale, ...
    'classOrder', string(primaryModel.ClassNames), ...
    'binaryLearners', numel(primaryModel.BinaryLearners), ...
    'binaryLoss', primaryModel.BinaryLoss, ...
    'parameters', bundle.parameters, ...
    'testWindows', testWindowCount, ...
    'primaryStaticCandidates', sum(trace.voteApplied), ...
    'voteChangedWindows', sum(trace.voteChangedPrediction), ...
    'primaryCorrect', sum(primaryPrediction == data.testLabels), ...
    'finalCorrect', sum(finalPrediction == data.testLabels), ...
    'focus', focusRecords, ...
    'disclaimer', disclaimer);

%% 特徴量のグループごとに、主 SVM がどれだけ頼っているかを調べる
% 既存の検証データだけを使い、モデルは学習し直さない
protocol = har.protocol(root, data);
validationFeatures = har.features( ...
    data.trainRaw(protocol.validationRows, :, :), bundle.spec.featureSet);
validationStandardized = standardize(validationFeatures, bundle.center, bundle.scale);
validationLabels = data.trainLabels(protocol.validationRows);
baselineAccuracy = mean( ...
    string(predict(primaryModel, validationStandardized)) == validationLabels);

% 32個の特徴量を、意味の近い列ごとに6グループへ分ける
groupColumns = {1:6, 7:12, 13:16, 17:20, 21:26, 27:32};
groupNames = ["axis_means", "axis_standard_deviations", "acceleration_magnitude", ...
    "gyroscope_magnitude", "axis_correlations", "band_power_fractions"];
permutationRepeats = 5;
permutationSeed = 51001;
validationCount = size(validationStandardized, 1);

sensitivity = struct([]);
rng(permutationSeed, 'twister');

for k = 1:numel(groupColumns)
    columns = groupColumns{k};
    accuracies = zeros(permutationRepeats, 1);

    % そのグループの列だけ行をでたらめに入れ替え、正解率の落ち幅を見る
    for repeat = 1:permutationRepeats
        perturbed = validationStandardized;
        shuffledRows = randperm(validationCount);
        perturbed(:, columns) = validationStandardized(shuffledRows, columns);
        accuracies(repeat) = mean( ...
            string(predict(primaryModel, perturbed)) == validationLabels);
    end

    record = struct('group', groupNames(k), ...
        'firstColumn', columns(1), ...
        'lastColumn', columns(end), ...
        'repeats', permutationRepeats, ...
        'baselineAccuracy', baselineAccuracy, ...
        'meanPermutedAccuracy', mean(accuracies), ...
        'meanAccuracyDropPercentagePoints', ...
        100 * (baselineAccuracy - mean(accuracies)), ...
        'stdAccuracyPercentagePoints', 100 * std(accuracies));
    sensitivity = appendRecord(sensitivity, record);
end

% 落ち幅の大きい (よく頼っている) グループから順に並べる
sensitivityTable = sortrows(struct2table(sensitivity), ...
    'meanAccuracyDropPercentagePoints', 'descend');
writetable(sensitivityTable, ...
    fullfile(outputFolder, 'primary-feature-group-sensitivity.csv'), ...
    'Encoding', 'UTF-8');

%% 根拠をまとめた JSON を書き出し、公表している数値と一致するか確かめる
evidence.primaryValidationGroupSensitivity = table2struct(sensitivityTable);
har.writeJson(fullfile(outputFolder, 'classification-evidence.json'), evidence);

expectedFinalCorrect = 201;
expectedPrimaryCorrect = 200;
expectedVoteChangedWindows = 1;
expectedVoteChangedRow = 155;
assert(evidence.finalCorrect == expectedFinalCorrect ...
    && evidence.primaryCorrect == expectedPrimaryCorrect ...
    && evidence.voteChangedWindows == expectedVoteChangedWindows);
assert(trace.row(trace.voteChangedPrediction) == expectedVoteChangedRow);

fprintf(['DECISION_EVIDENCE_EXPORTED: %d features, %d pairwise learners, ', ...
    '%d votes, %d changed prediction, %d/%d correct\n'], ...
    numel(featureNames), evidence.binaryLearners, evidence.primaryStaticCandidates, ...
    evidence.voteChangedWindows, evidence.finalCorrect, evidence.testWindows);

function standardized = standardize(features, center, scale)
% 学習時に求めた平均と標準偏差を使って標準化する。
standardized = (features - center) ./ scale;
end

function records = appendRecord(records, record)
% 記録を struct 配列の末尾に追加する。
% 空の struct 配列にはフィールドがなく、添字での代入ができないので場合分けする。
if isempty(records)
    records = record;
else
    records(end + 1) = record;
end
end
