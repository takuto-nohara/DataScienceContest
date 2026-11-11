function leaderboard = evaluate(root, ids, outputName)
% 保存済みのモデルを採点し、比較表とレポートを書き出す。
%
%   root       : リポジトリのフォルダ
%   ids        : 採点するモデルの名前。省略するとすべてのモデルを採点する
%   outputName : 結果を書き出す evaluation/ 配下のフォルダ名 (既定は 'latest')
%
% この関数は学習も調整もしない。保存済みのモデルをそのまま使う。
% 順位は検証データの正解率で決め、テストデータの正解率は参考として記録する。

if nargin < 2
    ids = string.empty;
end
if nargin < 3
    outputName = 'latest';
end
assert(~isempty(regexp(outputName, '^[A-Za-z0-9_-]+$', 'once')), ...
    'Invalid output name.');

data = har.loadData(root);
protocol = har.protocol(root, data);

folders = dir(fullfile(root, 'model', 'proto_*'));
folders = folders([folders.isdir]);
if ~isempty(ids)
    assert(all(ismember(string(ids), string({folders.name}))), ...
        'Unknown requested model ID.');
    folders = folders(ismember(string({folders.name}), string(ids)));
end
assert(~isempty(folders), 'No prototypes found.');

outputFolder = fullfile(root, 'evaluation', outputName);
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

validationRaw = data.trainRaw(protocol.validationRows, :, :);
validationLabels = data.trainLabels(protocol.validationRows);

records = struct([]);
details = struct([]);
for k = 1:numel(folders)
    folder = fullfile(folders(k).folder, folders(k).name);
    modelFile = fullfile(folder, 'trained.mat');
    bundle = loadVerifiedBundle(folder, modelFile, data, protocol);

    % 予測にかかる時間も、特徴量の計算を含めて測る。
    timer = tic;
    validationPrediction = har.predict(bundle, validationRaw);
    validationSeconds = toc(timer);

    timer = tic;
    testPrediction = har.predict(bundle, data.testRaw);
    testSeconds = toc(timer);

    validation = har.metrics(validationLabels, validationPrediction, data.classes);
    test = har.metrics(data.testLabels, testPrediction, data.classes);

    record = summarizeModel(bundle, modelFile, validation, test, ...
        validationSeconds, testSeconds);
    detail = struct('model', bundle.spec.id, 'validation', validation, 'test', test);
    if k == 1
        records = record;
        details = detail;
    else
        records(k) = record; %#ok<AGROW>
        details(k) = detail; %#ok<AGROW>
    end

    writeModelResults(outputFolder, bundle.spec.id, data, testPrediction, test);
    fprintf('%s: validation %.2f%%, external test %.2f%% (%d/%d)\n', ...
        bundle.spec.id, 100 * validation.accuracy, 100 * test.accuracy, ...
        test.correct, test.total);
end

% 検証データの正解率で順位を付ける。同点なら macro F1、次に名前の順。
leaderboard = sortrows(struct2table(records), ...
    {'validationAccuracy', 'validationMacroF1', 'model'}, ...
    {'descend', 'descend', 'ascend'});
modelCount = height(leaderboard);
leaderboard.validationRank = (1:modelCount)';

[~, testOrder] = sort(leaderboard.testAccuracy, 'descend');
testRank = zeros(modelCount, 1);
testRank(testOrder) = 1:modelCount;
leaderboard.testRank = testRank;
writetable(leaderboard, fullfile(outputFolder, 'leaderboard.csv'), ...
    'Encoding', 'UTF-8');

report = buildReport(leaderboard, details, data, protocol, testOrder);
har.writeJson(fullfile(outputFolder, 'results.json'), report);
har.renderReport(root, outputFolder, report);
end


function bundle = loadVerifiedBundle(folder, modelFile, data, protocol)
% モデルを読み込み、ほかのモデルと公平に比べられる状態かを確かめる。
% 条件が1つでも崩れていれば、採点せずにエラーで止める。
[~, modelName] = fileparts(folder);
assert(isfile(modelFile), 'Missing model: %s', modelFile);

% 保存後にファイルが書き換えられていないこと。
manifest = jsondecode(fileread(fullfile(folder, 'manifest.json')));
assert(strcmp(manifest.artifactSHA256, har.hashFile(modelFile)), ...
    'Model artifact checksum mismatch.');

saved = load(modelFile, 'bundle');
bundle = saved.bundle;

% 学習後に設定ファイルが変更されていないこと。
currentSpec = jsondecode(fileread(fullfile(folder, 'spec.json')));
assert(isequal(currentSpec, bundle.spec), 'Spec changed: retrain %s', modelName);

% 同じデータ、同じ分け方で学習したモデルであること。
assert(isequal(bundle.protocol.fingerprint, data.fingerprint), ...
    'Model/data fingerprint mismatch.');
assert(isequal(bundle.protocol.fitRows(:), protocol.fitRows(:)) ...
    && isequal(bundle.protocol.validationRows(:), protocol.validationRows(:)), ...
    'Models use different protocols.');

% 検証用の行が学習に混ざっていないこと。
assert(all(ismember(bundle.fitRows, protocol.fitRows)) ...
    && isempty(intersect(bundle.fitRows, protocol.validationRows)), ...
    'Training contamination.');
assert(isequal(string(bundle.classes(:)), data.classes), 'Class order mismatch.');
end


function record = summarizeModel(bundle, modelFile, validation, test, ...
    validationSeconds, testSeconds)
% 比較表の1行分をまとめる。

% テストラベルを見て調整したモデルには、その旨が spec に書いてある。
evaluationRole = "baseline_pre_tuning";
if isfield(bundle.spec, 'evaluationRole')
    evaluationRole = string(bundle.spec.evaluationRole);
end

% データ拡張をしたモデルでは、学習に使った件数が元の行数より多い。
fittingObservations = numel(bundle.fitRows);
if isfield(bundle, 'augmentedRows')
    fittingObservations = bundle.augmentedRows;
end

fileInfo = dir(modelFile);
record = struct( ...
    'model', string(bundle.spec.id), ...
    'algorithm', string(bundle.spec.algorithm), ...
    'evaluationRole', evaluationRole, ...
    'fittingObservations', fittingObservations, ...
    'featureSet', string(bundle.spec.featureSet), ...
    'features', numel(bundle.featureNames), ...
    'trainingRows', numel(bundle.fitRows), ...
    'validationAccuracy', validation.accuracy, ...
    'validationMacroF1', validation.macroF1, ...
    'testAccuracy', test.accuracy, ...
    'testMacroF1', test.macroF1, ...
    'testBalancedAccuracy', test.balancedAccuracy, ...
    'testCorrect', test.correct, ...
    'testTotal', test.total, ...
    'trainingSeconds', bundle.trainingSeconds, ...
    'validationSeconds', validationSeconds, ...
    'testSeconds', testSeconds, ...
    'modelBytes', fileInfo.bytes);
end


function writeModelResults(outputFolder, modelId, data, testPrediction, test)
% 1つのモデルについて、全試行の予測・混同行列・クラスごとの指標を書き出す。
rowNumbers = (1:numel(testPrediction))';
predictions = table(rowNumbers, data.testLabels, testPrediction, ...
    'VariableNames', {'row', 'truth', 'prediction'});
writetable(predictions, fullfile(outputFolder, [modelId '_predictions.csv']), ...
    'Encoding', 'UTF-8');

writematrix(test.confusion, fullfile(outputFolder, [modelId '_confusion.csv']), ...
    'Encoding', 'UTF-8');

perClass = table(data.classes, test.precision, test.recall, test.f1, test.support, ...
    'VariableNames', {'class', 'precision', 'recall', 'f1', 'support'});
writetable(perClass, fullfile(outputFolder, [modelId '_per_class.csv']), ...
    'Encoding', 'UTF-8');
end


function report = buildReport(leaderboard, details, data, protocol, testOrder)
% results.json とレポートに載せる内容をまとめる。結果の限界も必ず併記する。
testWindowCount = numel(data.testLabels);
limitations = { ...
    protocol.limitation, ...
    sprintf(['External test has %d windows only. Comparison is descriptive, ' ...
        'not an unbiased estimate after model selection.'], testWindowCount), ...
    ['Algorithms, feature sets, and training fractions differ; results do not ' ...
        'isolate the causal effect of algorithm choice.'], ...
    'Prototypes are not official submission bundles. Organizer verifier not supplied.'};

report = struct( ...
    'schemaVersion', 1, ...
    'protocolVersion', protocol.version, ...
    'fingerprint', data.fingerprint, ...
    'testWindows', testWindowCount, ...
    'validationWindows', numel(protocol.validationRows), ...
    'containsTestAwareModels', ...
        any(leaderboard.evaluationRole == "development_tuned"), ...
    'selectionMetric', 'validationAccuracy', ...
    'selectedModel', leaderboard.model(1), ...
    'bestObservedTestModel', leaderboard.model(testOrder(1)), ...
    'limitations', {limitations}, ...
    'leaderboard', table2struct(leaderboard), ...
    'details', details);

% テストラベルを見て調整したモデルが含まれるなら、その点数は独立した
% テスト結果ではないことを明記する。
if report.containsTestAwareModels
    report.limitations{end + 1} = ['The former external test labels guided ' ...
        'feature design, settings, augmentation and ensemble selection. ' ...
        'Its score is DEVELOPMENT accuracy, not independent test accuracy. ' ...
        'Validation was also inspected during this search.'];
end
end
