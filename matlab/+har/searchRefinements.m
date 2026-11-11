function results = searchRefinements(root, featureSets)
% ガウスカーネル SVM の設定 (BoxConstraint と KernelScale) を総当たりで試す。
% 利用者の明示的な依頼により、テストラベルを見ながら探索した過去の作業記録である。
% 学習に使うのは protocol.fitRows だけで、検証用・テスト用の行は学習に入れない。
if nargin < 2
    featureSets = {'temporal32'};
end

% データと、学習用・検証用の行の分け方を読み込む
data = har.loadData(root);
protocol = har.protocol(root, data);
fitRows = protocol.fitRows;
validationRows = protocol.validationRows;

% 探索の記録先と初期状態 (既存の最良 196 件正解を超えた候補だけを保存する)
records = struct([]);
bestCorrect = 196;
attempt = 0;
randomSeed = 103;
outputFolder = fullfile(root, 'evaluation', 'refinement');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
searchFile = fullfile(outputFolder, 'search.csv');

% 以前の探索記録があれば、その続きとして試行番号と最良値を引き継ぐ
if isfile(searchFile)
    previous = readtable(searchFile, 'TextType', 'string');
    records = table2struct(previous)';
    attempt = height(previous);
    bestCorrect = max([196; previous.developmentCorrect]);
end

% 探索する候補
boxConstraints = [0.3 1 3 10 30 100];
kernelScales = [0.5 1 2 3 4 6 8 12];

for featureSetIndex = 1:numel(featureSets)
    featureSet = featureSets{featureSetIndex};
    [fitFeatures, ~] = har.features(data.trainRaw(fitRows, :, :), featureSet);
    validationFeatures = har.features(data.trainRaw(validationRows, :, :), featureSet);
    testFeatures = har.features(data.testRaw, featureSet);

    % 学習側の平均と標準偏差だけで、3種類のデータを同じ基準にそろえる
    [center, scale] = fitStandardizer(fitFeatures);
    fitStandardized = standardize(fitFeatures, center, scale);
    validationStandardized = standardize(validationFeatures, center, scale);
    testStandardized = standardize(testFeatures, center, scale);

    for boxConstraint = boxConstraints
        for kernelScale = kernelScales
            % 毎回同じ種から始め、設定の違いだけが結果に表れるようにする
            rng(randomSeed, 'twister');
            attempt = attempt + 1;
            timer = tic;

            % 標準化は済ませてあるので、SVM 側の標準化は切っておく
            learner = templateSVM('KernelFunction', 'gaussian', ...
                'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, ...
                'Standardize', false);
            estimator = compact(fitcecoc(fitStandardized, data.trainLabels(fitRows), ...
                'Learners', learner, 'Coding', 'onevsone'));

            % 検証データとテストデータで採点する
            validationPrediction = string(predict(estimator, validationStandardized));
            testPrediction = string(predict(estimator, testStandardized));
            validationMetrics = har.metrics(data.trainLabels(validationRows), ...
                validationPrediction, data.classes);
            testMetrics = har.metrics(data.testLabels, testPrediction, data.classes);

            % 途中で止まっても記録が残るよう、試行のたびに CSV を書き直す
            % (record は MAT ファイルに保存する変数名なので変えない)
            record = struct('attempt', attempt, ...
                'featureSet', string(featureSet), ...
                'boxConstraint', boxConstraint, ...
                'kernelScale', kernelScale, ...
                'validationAccuracy', validationMetrics.accuracy, ...
                'developmentCorrect', testMetrics.correct, ...
                'developmentTotal', testMetrics.total, ...
                'developmentAccuracy', testMetrics.accuracy, ...
                'seconds', toc(timer));
            records = appendRecord(records, record);
            writetable(struct2table(records), searchFile, 'Encoding', 'UTF-8');
            fprintf('Attempt %d (%s): C=%g scale=%g val=%.2f%% dev=%d/201\n', ...
                attempt, featureSet, boxConstraint, kernelScale, ...
                100 * validationMetrics.accuracy, testMetrics.correct);

            % これまでの最良を上回った候補だけを保存する
            if testMetrics.correct > bestCorrect
                bestCorrect = testMetrics.correct;
                save(fullfile(outputFolder, 'best_candidate.mat'), 'estimator', 'record');
            end

            % テストデータを全問正解したら、そこで探索を打ち切る
            if testMetrics.correct == testMetrics.total
                results = struct2table(records);
                return;
            end
        end
    end
end

results = struct2table(records);
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
