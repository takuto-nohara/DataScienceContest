% コンテストで使う MATLAB 環境 (バージョン、ライセンス、各ツールボックス) を確かめる。
% 配布データには触れず、その場で作った合成データだけを使う。
root = fileparts(fileparts(mfilename('fullpath')));
outputFolder = fullfile(root, 'artifacts');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

% 画面に出した内容をログにも残す (途中でエラーになっても diary を確実に止める)
diary(fullfile(outputFolder, 'environment-check.log'));
cleanupDiary = onCleanup(@() diary('off'));

% MATLAB のバージョン
fprintf('MATLAB: %s\n', version);
assert(strcmp(version('-release'), '2026a'), 'Contest requires R2026a.');
assert(contains(version, 'Update 5'), 'Contest recommends Update 5.');
ver;

% 必要なライセンスを実際に取得できるか
licenseFeatures = {'MATLAB', 'Neural_Network_Toolbox', ...
    'Distrib_Computing_Toolbox', 'Statistics_Toolbox', 'Signal_Toolbox'};
for k = 1:numel(licenseFeatures)
    [isCheckedOut, message] = license('checkout', licenseFeatures{k});
    assert(isCheckedOut, 'License checkout failed for %s: %s', ...
        licenseFeatures{k}, message);
    fprintf('License checkout OK: %s\n', licenseFeatures{k});
end

% インストール先のフォルダを見るだけでなく、各ツールボックスの関数を実際に動かす
samplesPerWindow = 128;
classCount = 6;
trialsPerClass = 8;
channelCount = 6;
rng(224224);
syntheticFeatures = randn(classCount * trialsPerClass, channelCount);
syntheticLabels = categorical(repelem((1:classCount)', trialsPerClass));

% Statistics and Machine Learning Toolbox: 決定木の学習と予測
tree = fitctree(syntheticFeatures, syntheticLabels);
assert(numel(predict(tree, syntheticFeatures)) == size(syntheticFeatures, 1));

% Deep Learning Toolbox: dlarray の演算
deepLearningArray = dlarray(single(syntheticFeatures));
assert(isequal(size(extractdata(relu(deepLearningArray))), size(syntheticFeatures)));

% Signal Processing Toolbox: ピーク検出
sineWave = sin((0:samplesPerWindow - 1) * 0.3);
assert(numel(findpeaks(sineWave)) > 0);

% Parallel Computing Toolbox: 並列プールでの計算 (自分で作ったプールだけ片付ける)
workerCount = 2;
pool = gcp('nocreate');
createdPool = isempty(pool);
if createdPool
    pool = parpool('Processes', workerCount);
end
future = parfeval(pool, @sum, 1, 1:samplesPerWindow);
assert(fetchOutputs(future) == sum(1:samplesPerWindow));
if createdPool
    delete(pool);
end

% GPU は必須ではない。使えるかどうかを記録するだけで、なくても合格とする
try
    gpuCount = gpuDeviceCount('available');
    fprintf('Available GPUs: %d\n', gpuCount);
    if gpuCount > 0
        device = gpuDevice;
        fprintf('GPU: %s\n', device.Name);
        gpuValues = gpuArray(single(1:samplesPerWindow));
        assert(gather(sum(gpuValues)) == single(sum(1:samplesPerWindow)));
        fprintf('GPU computation OK\n');
    end
catch exception
    fprintf('Optional GPU check: %s\n', exception.message);
end

fprintf('ENVIRONMENT_CHECK_PASSED\n');
