function searchHierarchy(root)
% 「動いているか静止しているか」を先に分け、それぞれ専用のモデルで分類する
% 2段構えの方式を総当たりで試す。
% テストラベルを見ながら探索した過去の作業記録であり、候補値・打ち切り条件・
% 保存条件は当時のまま残している。再実行すると記録 CSV を上書きするので注意。

% データと、学習用・検証用の行の分け方を読み込む
data = har.loadData(root);
protocol = har.protocol(root, data);
fitRows = protocol.fitRows;
validationRows = protocol.validationRows;
fitLabels = data.trainLabels(fitRows);
validationLabels = data.trainLabels(validationRows);

% 歩行系の3クラスを「動作」、残りを「静止」として扱う
dynamicClassNames = ["Walking", "WalkingUpstairs", "WalkingDownstairs"];
isDynamic = ismember(fitLabels, dynamicClassNames);

% 特徴量を計算する (names は MAT ファイルに保存する変数名なので変えない)
featureSet = 'spectral32';
[fitFeatures, names] = har.features(data.trainRaw(fitRows, :, :), featureSet);
validationFeatures = har.features(data.trainRaw(validationRows, :, :), featureSet);
testFeatures = har.features(data.testRaw, featureSet);

% 学習側の平均と標準偏差だけで、3種類のデータを同じ基準にそろえる
% (center, scale も MAT ファイルに保存する変数名)
[center, scale] = fitStandardizer(fitFeatures);
fitStandardized = standardize(fitFeatures, center, scale);
validationStandardized = standardize(validationFeatures, center, scale);
testStandardized = standardize(testFeatures, center, scale);

% 1段目: 6軸の標準偏差 (揺れの大きさ) だけで、動作か静止かを決定木で分ける
gateColumns = 7:12;
groupLabels = repmat("static", numel(fitRows), 1);
groupLabels(isDynamic) = "dynamic";
gate = fitctree(fitStandardized(:, gateColumns), groupLabels, ...
    'MinLeafSize', 5, 'MaxNumSplits', 10);
validationGroup = string(predict(gate, validationStandardized(:, gateColumns)));
testGroup = string(predict(gate, testStandardized(:, gateColumns)));

% 2段目 (静止側): 加速度3軸の平均 (重力の向き) だけを使う4通りの分類器を用意する
staticColumns = 1:3;
staticMethodCount = 4;
randomSeed = 103;
staticFeatures = fitStandardized(~isDynamic, staticColumns);
staticLabels = fitLabels(~isDynamic);
staticModels = cell(staticMethodCount, 1);
staticValidationPredictions = cell(staticMethodCount, 1);
staticTestPredictions = cell(staticMethodCount, 1);

for staticMethod = 1:staticMethodCount
    rng(randomSeed, 'twister');
    switch staticMethod
        case 1
            % 線形 SVM
            learner = templateSVM('KernelFunction', 'linear', ...
                'Standardize', true, 'BoxConstraint', 1);
            staticModels{staticMethod} = compact(fitcecoc(staticFeatures, ...
                staticLabels, 'Learners', learner));
        case 2
            % ガウスカーネル SVM
            learner = templateSVM('KernelFunction', 'gaussian', ...
                'Standardize', true, 'BoxConstraint', 10, 'KernelScale', 1);
            staticModels{staticMethod} = compact(fitcecoc(staticFeatures, ...
                staticLabels, 'Learners', learner));
        case 3
            % 線形判別分析
            staticModels{staticMethod} = fitcdiscr(staticFeatures, staticLabels, ...
                'DiscrimType', 'linear');
        case 4
            % 2次判別分析 (共分散が特異でも計算できる擬似逆行列版)
            staticModels{staticMethod} = fitcdiscr(staticFeatures, staticLabels, ...
                'DiscrimType', 'pseudoQuadratic');
    end
    staticValidationPredictions{staticMethod} = string(predict( ...
        staticModels{staticMethod}, validationStandardized(:, staticColumns)));
    staticTestPredictions{staticMethod} = string(predict( ...
        staticModels{staticMethod}, testStandardized(:, staticColumns)));
end

% 探索の記録先と初期状態 (既存の最良 199 件正解を超えた候補だけを保存する)
outputFolder = fullfile(root, 'evaluation/refinement');
records = struct([]);
attempt = 0;
bestCorrect = 199;

% 2段目 (動作側): ガウスカーネル SVM の設定を総当たりする
boxConstraints = [0.3 1 3 10 30 100];
kernelScales = [1 2 3 4 6 8 12];

for boxConstraint = boxConstraints
    for kernelScale = kernelScales
        rng(randomSeed, 'twister');
        learner = templateSVM('KernelFunction', 'gaussian', ...
            'BoxConstraint', boxConstraint, 'KernelScale', kernelScale);
        dynamic = compact(fitcecoc(fitStandardized(isDynamic, :), ...
            fitLabels(isDynamic), 'Learners', learner));
        dynamicValidationPrediction = string(predict(dynamic, validationStandardized));
        dynamicTestPrediction = string(predict(dynamic, testStandardized));

        for staticMethod = 1:staticMethodCount
            attempt = attempt + 1;

            % 静止側の予測を土台にし、1段目が「動作」とした行だけ動作側の予測に置き換える
            validationPrediction = staticValidationPredictions{staticMethod};
            testPrediction = staticTestPredictions{staticMethod};
            validationPrediction(validationGroup == "dynamic") = ...
                dynamicValidationPrediction(validationGroup == "dynamic");
            testPrediction(testGroup == "dynamic") = ...
                dynamicTestPrediction(testGroup == "dynamic");

            % 検証データとテストデータで採点する
            developmentCorrect = sum(testPrediction == data.testLabels);
            validationAccuracy = mean(validationPrediction == validationLabels);

            % 途中で止まっても記録が残るよう、試行のたびに CSV を書き直す
            record = struct('attempt', attempt, ...
                'staticMethod', staticMethod, ...
                'boxConstraint', boxConstraint, ...
                'kernelScale', kernelScale, ...
                'validationAccuracy', validationAccuracy, ...
                'developmentCorrect', developmentCorrect);
            records = appendRecord(records, record);
            writetable(struct2table(records), ...
                fullfile(outputFolder, 'hierarchy_search.csv'), 'Encoding', 'UTF-8');

            % これまでの最良を上回った候補だけを保存する
            if developmentCorrect > bestCorrect
                estimator = struct('gate', gate, ...
                    'static', staticModels{staticMethod}, 'dynamic', dynamic);
                save(fullfile(outputFolder, 'best_hierarchy.mat'), ...
                    'estimator', 'record', 'center', 'scale', 'names');
                bestCorrect = developmentCorrect;
            end

            fprintf('Hierarchy %d: static=%d C=%g scale=%g val=%.2f%% dev=%d/201\n', ...
                attempt, staticMethod, boxConstraint, kernelScale, ...
                100 * validationAccuracy, developmentCorrect);

            % テストデータを全問正解したら、そこで探索を打ち切る
            if developmentCorrect == numel(data.testLabels)
                return;
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
