function cases = analyzeErrors(root)
% ベースライン (proto_v3) が誤った窓を1件ずつ調べる。
% スコアの差、学習データ中の最近傍、特徴量を差し替えたときの感度を
% CSV・PNG・JSON に書き出す。
baselineModel = 'proto_v3';

% 保存済みモデルを読み込み、評価用と学習用の特徴量を同じ尺度で標準化する
context.data = har.loadData(root);
saved = load(fullfile(root, 'model', baselineModel, 'trained.mat'), 'bundle');
context.bundle = saved.bundle;
bundle = context.bundle;
[context.testFeatures, context.featureNames] = ...
    har.features(context.data.testRaw, bundle.spec.featureSet);
[context.fitFeatures, ~] = ...
    har.features(context.data.trainRaw(bundle.fitRows,:,:), bundle.spec.featureSet);
context.testZ = (context.testFeatures - bundle.center) ./ bundle.scale;
context.fitZ = (context.fitFeatures - bundle.center) ./ bundle.scale;

% 予測が正解と違う窓だけを調べる対象にする
[predicted, context.scores] = predict(bundle.estimator, context.testZ);
context.predicted = string(predicted);
context.classes = string(bundle.estimator.ClassNames);
errorRows = find(context.predicted ~= context.data.testLabels);

context.outputFolder = fullfile(root, 'evaluation', 'error_analysis');
if ~isfolder(context.outputFolder)
    mkdir(context.outputFolder);
end

caseList = cell(1, numel(errorRows));
for k = 1:numel(errorRows)
    caseList{k} = analyzeCase(context, errorRows(k));
end
cases = struct([]);
if ~isempty(caseList)
    cases = [caseList{:}];
end

interpretation = ['Negative-loss scores are not probabilities. Feature ' ...
    'replacement is a sensitivity diagnostic, not a causal explanation or a ' ...
    'physically realizable counterfactual. Nearest neighbors do not themselves ' ...
    'determine the RBF SVM prediction.'];
har.writeJson(fullfile(context.outputFolder, 'baseline_errors.json'), ...
    struct('baseline', baselineModel, 'cases', cases, ...
    'interpretation', interpretation));
end

function caseSummary = analyzeCase(context, row)
% 誤分類された窓1件を調べ、表と図を保存して要約を返す。
topFeatureCount = 5;
bundle = context.bundle;
fitRows = bundle.fitRows;
fitLabels = context.data.trainLabels(fitRows);

truth = context.data.testLabels(row);
wrong = context.predicted(row);
actualClass = find(context.classes == truth);
wrongClass = find(context.classes == wrong);

% 正解クラスと誤答クラスそれぞれの学習データを取り出す
trueRows = find(fitLabels == truth);
wrongRows = find(fitLabels == wrong);
trueMedian = median(context.fitFeatures(trueRows,:), 1);
wrongMedian = median(context.fitFeatures(wrongRows,:), 1);

% 標準化後の距離で、学習データ全体と各クラス内の最近傍を探す
windowZ = context.testZ(row,:);
[distance, nearest] = sort(pdist2(windowZ, context.fitZ, 'euclidean'));
[~, nearTrue] = min(pdist2(windowZ, context.fitZ(trueRows,:), 'euclidean'));
[~, nearWrong] = min(pdist2(windowZ, context.fitZ(wrongRows,:), 'euclidean'));
trueReference = fitRows(trueRows(nearTrue));
wrongReference = fitRows(wrongRows(nearWrong));

% 負なら、正解クラスのスコアが誤答クラスより低かったことを表す
scoreGap = context.scores(row, actualClass) - context.scores(row, wrongClass);

% 特徴量を1つずつ正解クラスの中央値に差し替え、スコア差の変化を見る
featureCount = numel(context.featureNames);
counterfactual = repmat(windowZ, featureCount, 1);
for j = 1:featureCount
    counterfactual(j,j) = (trueMedian(j) - bundle.center(j)) / bundle.scale(j);
end
[counterfactualPrediction, counterfactualScore] = ...
    predict(bundle.estimator, counterfactual);
improvement = counterfactualScore(:, actualClass) ...
    - counterfactualScore(:, wrongClass) - scoreGap;

% スコア差を最も改善する特徴量が先頭に来るように並べて保存する
observed = context.testFeatures(row,:);
featureTable = table(context.featureNames', observed', trueMedian', wrongMedian', ...
    ((observed - trueMedian) ./ bundle.scale)', improvement, ...
    string(counterfactualPrediction), 'VariableNames', ...
    {'feature', 'observed', 'trueClassTrainingMedian', ...
    'predictedClassTrainingMedian', 'deviationInTrainingSD', ...
    'gapChangeIfSetToTrueMedian', 'counterfactualPrediction'});
featureTable = sortrows(featureTable, 'gapChangeIfSetToTrueMedian', 'descend');
writetable(featureTable, caseFile(context, row, 'features.csv'), ...
    'Encoding', 'UTF-8');

writeNeighborTable(context, row, nearest, distance);

caseSummary = struct('testRow', row, 'truth', truth, 'prediction', wrong, ...
    'classOrder', context.classes, ...
    'negativeLossScores', context.scores(row,:), ...
    'trueMinusPredictedScore', scoreGap, ...
    'nearestTrueTrainingRow', trueReference, ...
    'nearestWrongTrainingRow', wrongReference, ...
    'topSensitiveFeatures', table2struct(featureTable(1:topFeatureCount,:)));

plotSignalComparison(context, row, trueReference, wrongReference);

fprintf('Row %d: %s -> %s; score gap %.4f; most sensitive: %s\n', ...
    row, truth, wrong, scoreGap, featureTable.feature(1));
end

function writeNeighborTable(context, row, nearest, distance)
% 標準化後の距離が近い学習窓を、近い順に最大10件まで保存する。
maxNeighborCount = 10;
fitRows = context.bundle.fitRows;

count = min(maxNeighborCount, numel(nearest));
neighborRows = fitRows(nearest(1:count));
neighbors = table(neighborRows, context.data.trainLabels(neighborRows), ...
    distance(1:count)', ...
    'VariableNames', {'trainingRow', 'label', 'standardizedDistance'});
writetable(neighbors, caseFile(context, row, 'neighbors.csv'), ...
    'Encoding', 'UTF-8');
end

function plotSignalComparison(context, row, trueReference, wrongReference)
% 誤分類された窓の6信号を、正解クラス・誤答クラスの最近傍と重ねて保存する。
windowLength = 128;
sampleRateHz = 50;
channelCount = 6;
data = context.data;
truth = data.testLabels(row);
wrong = context.predicted(row);

fig = figure('Visible', 'off', 'Position', [50 50 1200 800], 'Color', 'white');
layout = tiledlayout(3, 2);
timeSeconds = (0:windowLength - 1) / sampleRateHz;
for channel = 1:channelCount
    nexttile;
    plot(timeSeconds, squeeze(data.testRaw(row,:,channel)), 'k', 'LineWidth', 1.3);
    hold on;
    plot(timeSeconds, squeeze(data.trainRaw(trueReference,:,channel)), 'b');
    plot(timeSeconds, squeeze(data.trainRaw(wrongReference,:,channel)), 'r:');

    % 先頭6個の特徴量は各信号の平均なので、その名前から信号名を取り出す
    channelName = strrep(char(context.featureNames(channel)), '_mean', '');
    title(channelName, 'Interpreter', 'none');
    xlabel('Seconds');
    grid on;
end
legend('Misclassified test window', 'Nearest true-class training window', ...
    'Nearest wrong-class training window', 'Location', 'best');
title(layout, sprintf('Row %d: true %s, predicted %s', row, truth, wrong), ...
    'Interpreter', 'none');

exportgraphics(fig, caseFile(context, row, 'signals.png'), 'Resolution', 130);
close(fig);
end

function file = caseFile(context, row, suffix)
% 窓ごとの出力ファイル名 (row_008_features.csv など) を組み立てる。
file = fullfile(context.outputFolder, sprintf('row_%03d_%s', row, suffix));
end
