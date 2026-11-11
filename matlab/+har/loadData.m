function data = loadData(root)
% 主催者が配布した学習用・テスト用のセンサデータを読み込む。
%
%   root : リポジトリのフォルダ
%   data : 次のフィールドを持つ構造体
%          trainRaw, testRaw       N×128×6 の波形 (加速度 x/y/z、角速度 x/y/z)
%          trainLabels, testLabels N×1 の行動名
%          classes                 6つの行動名
%          fingerprint             2つのファイルの SHA-256
%
% すべてのモデルがこの関数を通して同じデータを読む。配布ファイルに入っている
% 計算済みの特徴量は使わず、波形から har.features で計算し直す。

dataFolder = fullfile(root, 'data', 'prestudy', 'Rawdata Files');
trainFile = fullfile(dataFolder, 'rawSensorData_train.mat');
testFile = fullfile(dataFolder, 'rawSensorData_test.mat');
assert(isfile(trainFile) && isfile(testFile), ...
    'Download organizer data into data/prestudy first.');

train = load(trainFile);
test = load(testFile);

sensorVariables = {'total_acc_x', 'total_acc_y', 'total_acc_z', ...
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z'};
samplesPerWindow = 128;

data.classes = ["Walking"; "WalkingUpstairs"; "WalkingDownstairs"; ...
    "Sitting"; "Standing"; "Laying"];
data.trainLabels = string(train.trainActivity(:));
data.testLabels = string(test.testActivity(:));
assert(all(ismember(data.trainLabels, data.classes)) ...
    && all(ismember(data.testLabels, data.classes)), 'Unknown activity label.');

trainCount = numel(data.trainLabels);
testCount = numel(data.testLabels);
data.trainRaw = zeros(trainCount, samplesPerWindow, numel(sensorVariables));
data.testRaw = zeros(testCount, samplesPerWindow, numel(sensorVariables));

for k = 1:numel(sensorVariables)
    % テスト用ファイルの変数名には、末尾に _test が付いている。
    trainSignal = train.(sensorVariables{k});
    testSignal = test.([sensorVariables{k} '_test']);

    assert(isequal(size(trainSignal), [trainCount, samplesPerWindow]) ...
        && isequal(size(testSignal), [testCount, samplesPerWindow]), ...
        'Invalid sensor shape.');
    assert(isreal(trainSignal) && isreal(testSignal) ...
        && all(isfinite(trainSignal(:))) && all(isfinite(testSignal(:))), ...
        'Invalid sensor values.');

    data.trainRaw(:, :, k) = trainSignal;
    data.testRaw(:, :, k) = testSignal;
end

% ファイルが差し替わったことを検出できるよう、ハッシュ値を控えておく。
data.fingerprint = struct( ...
    'train', har.hashFile(trainFile), ...
    'test', har.hashFile(testFile));
end
