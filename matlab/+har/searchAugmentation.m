function searchAugmentation(root)
% 振幅を変えた波形で学習データを水増しし、ガウスカーネル SVM の設定を総当たりで試す。
% テストラベルを見ながら探索した過去の作業記録であり、候補値・打ち切り条件・
% 保存条件は当時のまま残している。再実行すると記録 CSV を上書きするので注意。

% データと、学習用・検証用の行の分け方を読み込む
data = har.loadData(root);
protocol = har.protocol(root, data);
fitRows = protocol.fitRows;
validationRows = protocol.validationRows;
validationLabels = data.trainLabels(validationRows);

% 探索の記録先と初期状態 (既存の最良 199 件正解を超えた候補だけを保存する)
outputFolder = fullfile(root, 'evaluation/refinement');
records = struct([]);
attempt = 0;
bestCorrect = 199;
randomSeed = 103;

% 探索する候補。振幅の倍率は 1 (元の波形) を必ず含む
gainSets = {[1 0.8 1.2], [1 0.6 1.4], [1 0.6 0.8 1.2 1.4]};
featureSets = {'spectral32', 'shape32', 'temporal32'};
boxConstraints = [1 3 10 30];
kernelScales = [3 4 6 8 12];

for featureSetIndex = 1:numel(featureSets)
    featureSet = featureSets{featureSetIndex};
    validationFeatures = har.features(data.trainRaw(validationRows, :, :), featureSet);
    testFeatures = har.features(data.testRaw, featureSet);

    for gainSetIndex = 1:numel(gainSets)
        % 水増しするのは学習用の行だけ。検証・テストの波形には手を加えない
        [augmentedRaw, augmentedLabels] = har.augment(data.trainRaw(fitRows, :, :), ...
            data.trainLabels(fitRows), gainSets{gainSetIndex});
        fitFeatures = har.features(augmentedRaw, featureSet);

        % 学習側の平均と標準偏差だけで、3種類のデータを同じ基準にそろえる
        % (center, scale は MAT ファイルに保存する変数名なので変えない)
        [center, scale] = fitStandardizer(fitFeatures);
        fitStandardized = standardize(fitFeatures, center, scale);
        validationStandardized = standardize(validationFeatures, center, scale);
        testStandardized = standardize(testFeatures, center, scale);

        for boxConstraint = boxConstraints
            for kernelScale = kernelScales
                attempt = attempt + 1;

                % 毎回同じ種から始め、設定の違いだけが結果に表れるようにする
                rng(randomSeed, 'twister');
                learner = templateSVM('KernelFunction', 'gaussian', ...
                    'KernelScale', kernelScale, 'BoxConstraint', boxConstraint);
                estimator = compact(fitcecoc(fitStandardized, augmentedLabels, ...
                    'Learners', learner, 'Coding', 'onevsone'));

                % 検証データとテストデータで採点する
                validationPrediction = string(predict(estimator, validationStandardized));
                testPrediction = string(predict(estimator, testStandardized));
                developmentCorrect = sum(testPrediction == data.testLabels);
                validationAccuracy = mean(validationPrediction == validationLabels);

                % 途中で止まっても記録が残るよう、試行のたびに CSV を書き直す
                record = struct('attempt', attempt, ...
                    'featureSet', string(featureSet), ...
                    'gainSet', gainSetIndex, ...
                    'augmentedRows', numel(augmentedLabels), ...
                    'boxConstraint', boxConstraint, ...
                    'kernelScale', kernelScale, ...
                    'validationAccuracy', validationAccuracy, ...
                    'developmentCorrect', developmentCorrect);
                records = appendRecord(records, record);
                writetable(struct2table(records), ...
                    fullfile(outputFolder, 'augmentation_search.csv'), ...
                    'Encoding', 'UTF-8');

                % これまでの最良を上回った候補だけを保存する
                if developmentCorrect > bestCorrect
                    gains = gainSets{gainSetIndex};
                    save(fullfile(outputFolder, 'best_augmented.mat'), ...
                        'estimator', 'record', 'center', 'scale', 'gains');
                    bestCorrect = developmentCorrect;
                end

                fprintf(['Augment %d (%s gains%d): C=%g scale=%g ', ...
                    'val=%.2f%% dev=%d/201\n'], ...
                    attempt, featureSet, gainSetIndex, boxConstraint, kernelScale, ...
                    100 * validationAccuracy, developmentCorrect);

                % テストデータを全問正解したら、そこで探索を打ち切る
                if developmentCorrect == numel(data.testLabels)
                    return;
                end
            end
        end
    end
end
end

function [center, scale] = fitStandardizer(features)
% 標準化に使う平均と標準偏差を求める。
% ばらつきがほぼ 0 の列は、0 で割らないよう標準偏差を 1 とみなす。
minimumScale = 1e-12;
center = mean(features);
scale = std(features);
scale(scale < minimumScale) = 1;
end

function standardized = standardize(features, center, scale)
% 学習側で求めた平均と標準偏差を使って標準化する。
standardized = (features - center) ./ scale;
end

function records = appendRecord(records, record)
% 試行の記録を struct 配列の末尾に追加する。
% 空の struct 配列にはフィールドがなく、添字での代入ができないので場合分けする。
if isempty(records)
    records = record;
else
    records(end + 1) = record;
end
end
