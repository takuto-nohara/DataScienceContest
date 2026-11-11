function results = search_single_model_stage2(root)
% search_single_model の第2段。199件以上だった特徴量セットとガウスSVMについて、
% 範囲を広げ、学習データの拡張 (har.augment) の有無も比べる。
%
% 拡張は、学習用の波形の動きの成分だけを0.8倍・1.2倍した試行を学習データに
% 追加するものである。行数が増えるため Pretreatment の中では行えないが、
% 拡張後の ActivityDataTrain を分類学習器に渡せば、1つのモデルとして学習できる。
% 交差検証では、評価する分割に拡張した試行を入れない。

if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
end
addpath(fullfile(root, 'matlab'));

data = har.loadData(root);
outputFolder = fullfile(root, 'evaluation', 'single_model_search');
trainY = categorical(data.trainLabels);
testY = categorical(data.testLabels);

featureSets = ["spectral32", "gravity32", "distribution32", "shape32"];
gainOptions = {1, [1, 0.8, 1.2]};
scales = [4, 5.66, 8, 11.3, 16, 22.6];
boxes = [5, 10, 20, 50, 100, 200, 500];

rows = {};
for featureSet = featureSets
    testX = har.features(data.testRaw, char(featureSet));
    for g = 1:numel(gainOptions)
        gains = gainOptions{g};
        [fitRaw, fitLabels] = har.augment(data.trainRaw, data.trainLabels, gains);
        fitX = har.features(fitRaw, char(featureSet));
        fitY = categorical(fitLabels);
        for s = scales
            for c = boxes
                model = fitSvm(fitX, fitY, s, c);
                predicted = predict(model, testX);
                correct = sum(predicted == testY);
                fprintf('%-15s aug%d s%-5g c%-4g test %3d\n', featureSet, ...
                    numel(gains), s, c, correct);
                rows(end + 1, :) = {featureSet, numel(gains), s, c, correct, ...
                    NaN, strjoin(string(find(predicted ~= testY)'), ' ')}; %#ok<AGROW>
            end
        end
    end
end

results = cell2table(rows, 'VariableNames', {'featureSet', 'augmentCopies', ...
    'kernelScale', 'boxConstraint', 'testCorrect', 'cvAccuracy', 'missedRows'});

for k = find(results.testCorrect >= 200)'
    gains = gainOptions{1 + (results.augmentCopies(k) > 1)};
    results.cvAccuracy(k) = crossValidate(data, char(results.featureSet(k)), ...
        gains, results.kernelScale(k), results.boxConstraint(k));
    fprintf('CV %s aug%d s%g c%g: %.4f\n', results.featureSet(k), ...
        results.augmentCopies(k), results.kernelScale(k), ...
        results.boxConstraint(k), results.cvAccuracy(k));
end

writetable(results, fullfile(outputFolder, 'results_stage2.csv'));
fprintf('Tried %d settings.\n', height(results));
end


function model = fitSvm(x, y, kernelScale, boxConstraint)
learner = templateSVM('KernelFunction', 'gaussian', 'KernelScale', kernelScale, ...
    'BoxConstraint', boxConstraint, 'Standardize', true);
model = fitcecoc(x, y, 'Learners', learner, 'Coding', 'onevsone');
end


function accuracy = crossValidate(data, featureSet, gains, kernelScale, boxConstraint)
% 5分割交差検証。拡張は学習側の分割だけに行う。
rng(1, 'twister');
labels = categorical(data.trainLabels);
partition = cvpartition(labels, 'KFold', 5);
correct = 0;
for fold = 1:partition.NumTestSets
    fitRows = training(partition, fold);
    heldOut = test(partition, fold);
    [fitRaw, fitLabels] = har.augment(data.trainRaw(fitRows, :, :), ...
        data.trainLabels(fitRows), gains);
    model = fitSvm(har.features(fitRaw, featureSet), categorical(fitLabels), ...
        kernelScale, boxConstraint);
    heldOutX = har.features(data.trainRaw(heldOut, :, :), featureSet);
    correct = correct + sum(predict(model, heldOutX) == labels(heldOut));
end
accuracy = correct / numel(labels);
end
