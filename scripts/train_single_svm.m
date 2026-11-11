function info = train_single_svm(root)
% 多数決を使わない1つのガウスSVMを、分類学習器と同じ手順で学習し、評価する。
%
%   root : リポジトリのフォルダ (省略時はこのファイルの1つ上)
%   info : 設定と正解率をまとめた構造体 (model/single_svm_v1/model-info.json にも保存)
%
% 練習2 (Excercise2_HumanActivityClassification_ML_ja.mlx) の流れに合わせている。
%   1. rawSensorData_train.mat を読み、前処理関数で ActivityDataTrain を作る
%   2. 学習用の試行を拡張し、ActivityDataTrain と trainActivity に追加する
%   3. 分類学習器と同じ設定のSVMを学習する (5分割交差検証で検証する)
%   4. rawSensorData_test.mat を読み、同じ前処理関数で ActivityDataTest を作り、
%      テストする (分類学習器の「すべてテスト」に当たる)
%   5. 分類学習器の「モデルのエクスポート」と同じ形の trainedModel を保存する
%
% 分類学習器で同じモデルを作るときの設定は presentation/classification_learner.md
% に書いてある。

if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
end
addpath(fullfile(root, 'matlab'));
submissionFolder = fullfile(root, 'submission', 'gravity32_single_svm');
addpath(submissionFolder);
cleanup = onCleanup(@() rmpath(submissionFolder));

dataFolder = fullfile(root, 'data', 'prestudy', 'Rawdata Files');
outputFolder = fullfile(root, 'model', 'single_svm_v1');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

settings = struct( ...
    'featureSet', 'gravity32', ...
    'augmentationGains', [1, 0.8, 1.2], ...
    'kernelFunction', 'gaussian', ...
    'kernelScale', 8, ...
    'boxConstraint', 50, ...
    'standardize', true, ...
    'coding', 'onevsone', ...
    'crossValidationFolds', 5, ...
    'crossValidationSeed', 1);

%% 1. 学習用データを読み、前処理関数で特徴量を作る
trainFile = load(fullfile(dataFolder, 'rawSensorData_train.mat'));
trainRaw = stackSignals(trainFile, '');
trainActivity = categorical(trainFile.trainActivity);

%% 2. 学習用の試行を拡張する (動きの成分だけを0.8倍・1.2倍した試行を追加)
[augmentedRaw, augmentedLabels] = har.augment(trainRaw, string(trainActivity), ...
    settings.augmentationGains);
ActivityDataTrain = pretreat(augmentedRaw);
trainActivityAugmented = categorical(augmentedLabels, categories(trainActivity));

%% 3. SVMを学習し、5分割交差検証で検証する
% 拡張した試行が、元の試行と別の分割に入ると検証が甘くなる。そこで分割は
% 元の2941件で決め、学習側の分割だけを拡張する。
rng(settings.crossValidationSeed, 'twister');
partition = cvpartition(trainActivity, 'KFold', settings.crossValidationFolds);
validationPredictions = trainActivity;
for fold = 1:partition.NumTestSets
    fitRows = training(partition, fold);
    heldOut = test(partition, fold);
    [foldRaw, foldLabels] = har.augment(trainRaw(fitRows, :, :), ...
        string(trainActivity(fitRows)), settings.augmentationGains);
    foldModel = fitSvm(pretreat(foldRaw), ...
        categorical(foldLabels, categories(trainActivity)), settings);
    validationPredictions(heldOut) = predict(foldModel, pretreat(trainRaw(heldOut, :, :)));
end
validationAccuracy = mean(validationPredictions == trainActivity);

classificationSVM = fitSvm(ActivityDataTrain, trainActivityAugmented, settings);

%% 4. テスト用データで評価する (分類学習器の「すべてテスト」)
testFile = load(fullfile(dataFolder, 'rawSensorData_test.mat'));
% 配布ファイルの ActivityDataTest は練習1の12列なので、前処理関数で作り直す。
ActivityDataTest = pretreat(stackSignals(testFile, '_test'));
testActivity = categorical(testFile.testActivity, categories(trainActivity));
testPredictions = predict(classificationSVM, ActivityDataTest);
testCorrect = sum(testPredictions == testActivity);

%% 5. 分類学習器の「モデルのエクスポート」と同じ形で保存する
trainedModel = struct();
trainedModel.predictFcn = @(x) predict(classificationSVM, x);
trainedModel.ClassificationSVM = classificationSVM;
trainedModel.About = ['Gaussian SVM (one-vs-one) on 32 Pretreatment features ' ...
    '(gravity32), trained on 2941 organizer training trials plus amplitude-' ...
    'augmented copies (gains 0.8, 1.2).'];
trainedModel.HowToPredict = ['yfit = trainedModel.predictFcn(X), where X is ' ...
    'the N x 32 output of Pretreatment_MATLABcontest_Advanced.'];
save(fullfile(outputFolder, 'trainedModel.mat'), 'trainedModel');

info = struct( ...
    'id', 'single_svm_v1', ...
    'settings', settings, ...
    'trainingTrials', numel(trainActivity), ...
    'trainingRowsAfterAugmentation', size(ActivityDataTrain, 1), ...
    'validationAccuracy', validationAccuracy, ...
    'testCorrect', testCorrect, ...
    'testTotal', numel(testActivity), ...
    'testMissedRows', find(testPredictions ~= testActivity), ...
    'testConfusion', confusionmat(testActivity, testPredictions, ...
        'Order', categories(trainActivity)), ...
    'classOrder', {categories(trainActivity)}, ...
    'selection', ['Chosen after 858 settings were scored on the 201 development ' ...
        'labels (scripts/search_single_model*.m). 201/201 is a selected result, ' ...
        'not an independent test.'], ...
    'matlabVersion', version);
har.writeJson(fullfile(outputFolder, 'model-info.json'), info);

fprintf('single_svm_v1: validation %.2f%%, test %d/%d\n', ...
    100 * validationAccuracy, testCorrect, numel(testActivity));
end


function raw = stackSignals(source, suffix)
% 6つのセンサ変数を N×128×6 の配列にまとめる。
names = {'total_acc_x', 'total_acc_y', 'total_acc_z', ...
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z'};
raw = zeros([size(source.([names{1} suffix])), numel(names)]);
for k = 1:numel(names)
    raw(:, :, k) = source.([names{k} suffix]);
end
end


function features = pretreat(raw)
% 提出用の前処理関数を、N×128×6 の配列に対して呼ぶ。
features = Pretreatment_MATLABcontest_Advanced(raw(:, :, 1), raw(:, :, 2), ...
    raw(:, :, 3), raw(:, :, 4), raw(:, :, 5), raw(:, :, 6));
end


function model = fitSvm(x, y, settings)
% 分類学習器の「SVM」で、カーネル関数をガウス、カーネルスケールと
% ボックス制約を手動、標準化をオン、多クラスを1対1にした場合と同じ設定。
learner = templateSVM('KernelFunction', settings.kernelFunction, ...
    'KernelScale', settings.kernelScale, ...
    'BoxConstraint', settings.boxConstraint, ...
    'Standardize', settings.standardize);
model = fitcecoc(x, y, 'Learners', learner, 'Coding', settings.coding, ...
    'ClassNames', categorical(categories(y), categories(y)));
end
