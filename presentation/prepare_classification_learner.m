%% 分類学習器に渡す変数をワークスペースに用意する
% 練習2 (Excercise2_HumanActivityClassification_ML_ja.mlx) と同じ変数名で、
% 32列の特徴量を作る。実行後に classificationLearner を開き、
% presentation/classification_learner.md の手順で学習とテストを行う。
%
% できる変数:
%   ActivityDataTrain : 学習用の特徴量 (2941試行 + 拡張した試行) × 32列
%   trainActivity     : ActivityDataTrain の各行の行動名
%   ActivityDataTest  : テスト用201試行 × 32列 (前処理関数で作り直したもの)
%   testActivity      : テスト用の行動名
%
% useAugmentation を false にすると、拡張なしの2941行になる。

useAugmentation = true;
augmentationGains = [1, 0.8, 1.2];

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repositoryRoot, 'matlab'));
addpath(fullfile(repositoryRoot, 'submission', 'gravity32_single_svm'));
dataFolder = fullfile(repositoryRoot, 'data', 'prestudy', 'Rawdata Files');

%% 学習用データ (練習1の「データのロード」と同じファイル)
load(fullfile(dataFolder, 'rawSensorData_train.mat'));
trainRaw = cat(3, total_acc_x, total_acc_y, total_acc_z, ...
    body_gyro_x, body_gyro_y, body_gyro_z);
trainActivity = categorical(trainActivity);

if useAugmentation
    % 動きの成分だけを0.8倍・1.2倍した試行を追加する (重力の向きは変えない)。
    [trainRaw, augmentedLabels] = har.augment(trainRaw, string(trainActivity), ...
        augmentationGains);
    trainActivity = categorical(augmentedLabels, categories(trainActivity));
end

ActivityDataTrain = Pretreatment_MATLABcontest_Advanced(trainRaw(:, :, 1), ...
    trainRaw(:, :, 2), trainRaw(:, :, 3), trainRaw(:, :, 4), ...
    trainRaw(:, :, 5), trainRaw(:, :, 6));

%% テスト用データ (練習2の「トレーニング済みモデルの評価」と同じファイル)
% このファイルの ActivityDataTest は練習1の12列なので、前処理関数で作り直す。
load(fullfile(dataFolder, 'rawSensorData_test.mat'));
ActivityDataTest = Pretreatment_MATLABcontest_Advanced(total_acc_x_test, ...
    total_acc_y_test, total_acc_z_test, body_gyro_x_test, body_gyro_y_test, ...
    body_gyro_z_test);
testActivity = categorical(testActivity, categories(trainActivity));

clear total_acc_* body_gyro_* trainRaw augmentedLabels dataFolder
fprintf('ActivityDataTrain %d x %d, ActivityDataTest %d x %d\n', ...
    size(ActivityDataTrain), size(ActivityDataTest));
