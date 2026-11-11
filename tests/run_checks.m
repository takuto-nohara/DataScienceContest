% 特徴量・水増し・評価指標・学習手順・保存済みモデルの基本的な性質を確かめる。
% 学習はせず、乱数で作った波形と保存済みモデルだけを使うので短時間で終わる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% 乱数で作った検査用の波形 (9試行 × 128点 × 6軸)
trialCount = 9;
samplesPerWindow = 128;
channelCount = 6;
rng(17);
raw = randn(trialCount, samplesPerWindow, channelCount);
tolerance = 1e-10;

%% 特徴量: 列数と名前、行ごとの独立性
% 特徴量セットの名前と、期待する特徴量の数
featureSets = { ...
    'stats12', 12; ...
    'temporal32', 32; ...
    'accel20', 20; ...
    'stable26', 26; ...
    'gravity32', 32; ...
    'spectral32', 32; ...
    'distribution32', 32; ...
    'window32', 32; ...
    'shape32', 32};

for k = 1:size(featureSets, 1)
    featureSet = featureSets{k, 1};
    featureCount = featureSets{k, 2};

    [features, names] = har.features(raw, featureSet);
    assert(isequal(size(features), [trialCount featureCount]) ...
        && numel(unique(names)) == featureCount);

    % 同じ試行は、単独でも、別の組み合わせや並び順でも、同じ値にならなければならない
    singleRow = 4;
    assert(max(abs(features(singleRow, :) ...
        - har.features(raw(singleRow, :, :), featureSet)), [], 'all') < tolerance);
    reorderedRows = [8 2];
    assert(max(abs(features(reorderedRows, :) ...
        - har.features(raw(reorderedRows, :, :), featureSet)), [], 'all') < tolerance);

    % 全く動きのない波形でも NaN や Inf を出さない
    zeroFeatures = har.features(zeros(1, samplesPerWindow, channelCount), featureSet);
    assert(all(isfinite(zeroFeatures)));
end

%% 水増し: ラベルの並びと、重力の向きの保存
gains = [1 0.8 1.2];
labels = string((1:trialCount)');
accelerometerChannels = 1:3;
[expanded, expandedLabels] = har.augment(raw, labels, gains);
assert(isequal(expandedLabels, repmat(labels, numel(gains), 1)));

% 振幅を変えても、加速度の平均 (重力の推定値) は元の波形と同じでなければならない
for block = 0:numel(gains) - 1
    blockRows = block * trialCount + (1:trialCount);
    assert(max(abs(mean(expanded(blockRows, :, accelerometerChannels), 2) ...
        - mean(raw(:, :, accelerometerChannels), 2)), [], 'all') < 1e-12, ...
        'Augmentation changed estimated gravity.');
end

%% センサーの分離: 加速度だけの特徴量は、ジャイロの値に左右されない
gyroscopeChannels = 4:6;
accelFeatures = har.features(raw, 'accel20');
raw(:, :, gyroscopeChannels) = 1000 * randn(trialCount, samplesPerWindow, 3);
assert(isequal(accelFeatures, har.features(raw, 'accel20')), ...
    'Accelerometer-only features used gyroscope data.');

%% 評価指標: 手計算できる小さな例と一致するか
result = har.metrics(["a"; "a"; "b"; "b"], ["a"; "b"; "b"; "b"], ["a"; "b"]);
assert(isequal(result.confusion, [1 1; 0 2]) && result.accuracy == 0.75);
assert(abs(result.macroF1 - (2/3 + 0.8) / 2) < 1e-12 ...
    && result.balancedAccuracy == 0.75);

%% 不正な入力: 黙って受け入れず、エラーにする
% 未知のラベル、NaN を含む波形、長さの足りない波形
assert(isRejected(@() har.metrics("a", "unknown", ["a"; "b"])));
assert(isRejected(@() har.features(nan(1, samplesPerWindow, channelCount), 'stats12')));
assert(isRejected( ...
    @() har.features(zeros(1, samplesPerWindow - 1, channelCount), 'stats12')));

%% 学習手順: 学習用と検証用の行が重なっていないか
data = har.loadData(root);
protocol = har.protocol(root, data);
assert(isempty(intersect(protocol.fitRows, protocol.validationRows)));

% 存在しないモデル ID は、その旨のエラーで拒否されなければならない
[~, message] = isRejected(@() har.evaluate(root, {'proto_missing'}));
assert(contains(message, 'Unknown requested model ID'), ...
    'Evaluator silently accepted an unknown model ID.');

%% 保存済みモデル: 学習に使った行、1件ずつの予測、標準化の基準
modelFiles = dir(fullfile(root, 'model', 'proto_*', 'trained.mat'));
batchSize = 8;

for k = 1:numel(modelFiles)
    saved = load(fullfile(modelFiles(k).folder, modelFiles(k).name), 'bundle');
    bundle = saved.bundle;

    % 学習に使った行は、すべて学習用の行でなければならない
    assert(all(ismember(bundle.fitRows, protocol.fitRows)));

    % まとめて予測しても、1件ずつ予測しても、同じ結果にならなければならない
    batchPrediction = har.predict(bundle, data.testRaw(1:batchSize, :, :));
    for row = 1:batchSize
        assert(batchPrediction(row) == har.predict(bundle, data.testRaw(row, :, :)));
    end

    % 標準化の平均は、学習に使った波形 (水増し後) だけから求めたものでなければならない
    fitRaw = data.trainRaw(bundle.fitRows, :, :);
    if isfield(bundle.spec, 'augmentationGains')
        [fitRaw, ~] = har.augment(fitRaw, data.trainLabels(bundle.fitRows), ...
            bundle.spec.augmentationGains);
    end
    [fitFeatures, ~] = har.features(fitRaw, bundle.spec.featureSet);
    assert(max(abs(mean(fitFeatures, 1) - bundle.center)) < tolerance, ...
        'Scaler not fitted on training rows.');
end

%% 多数決: 静止クラスにだけ適用され、同数のときは主モデルの予測を残す
primary = ["Standing"; "Walking"; "Sitting"];
votes = [ ...
    "Standing", "Sitting", "Sitting"; ...
    "Walking", "Sitting", "Sitting"; ...
    "Sitting", "Standing", "Laying"];
assert(isequal(har.staticVote(primary, votes), ["Sitting"; "Walking"; "Sitting"]), ...
    'Static gate or tie behavior failed.');

fprintf(['CHECKS_PASSED: features, row independence, sensor isolation, metrics, ', ...
    'invalid-input rejection, protocol, saved-model inference (%d models)\n'], ...
    numel(modelFiles));

function [rejected, message] = isRejected(action)
% 渡された処理がエラーになることを確かめ、そのエラーメッセージも返す。
% エラーにならなかった場合は rejected が false、message が空になる。
rejected = false;
message = '';
try
    action();
catch exception
    rejected = true;
    message = exception.message;
end
end
