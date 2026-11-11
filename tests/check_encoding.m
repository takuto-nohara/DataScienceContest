% このファイル自体を UTF-8 で保存し、MATLAB が日本語を正しく読み書きできることを確かめる。
% あわせて、モデルのバイナリファイルが書き換えられていないことも確かめる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% ソースコード中の日本語が、画面表示と UTF-8 への往復変換で壊れない
expected = '日本語の表示確認';
fprintf('%s\n', expected);
bytes = unicode2native(expected, 'UTF-8');
assert(isequal(native2unicode(bytes, 'UTF-8'), expected));

% UTF-8 で保存した日本語のレポートを正しく読める
report = fileread(fullfile(root, 'reports', 'classification-criteria-ja.md'));
assert(contains(report, '行動認識モデルは何を基準に分類しているか'));

% 日本語を含む表を CSV に書き、読み戻せる
csvFile = fullfile(root, 'artifacts', 'encoding-japanese.csv');
fixture = table(["歩行"; "立位"], ["正解"; "正解"], ...
    'VariableNames', {'activity', 'result'});
writetable(fixture, csvFile, 'Encoding', 'UTF-8');
assert(contains(fileread(csvFile), '歩行'));

% 日本語を含む JSON を書き、読み戻せる
jsonFile = fullfile(root, 'artifacts', 'encoding-japanese.json');
har.writeJson(jsonFile, struct('message', expected));
decoded = jsondecode(fileread(jsonFile));
assert(strcmp(decoded.message, expected));

% モデルのバイナリは読み込めて、記録したハッシュ値と一致する
modelFiles = dir(fullfile(root, 'model', 'proto_*', 'trained.mat'));
for k = 1:numel(modelFiles)
    modelFolder = modelFiles(k).folder;
    modelFile = fullfile(modelFolder, modelFiles(k).name);

    loaded = load(modelFile, 'bundle');
    assert(isstruct(loaded.bundle) && ~isempty(loaded.bundle.featureNames));

    manifest = jsondecode(fileread(fullfile(modelFolder, 'manifest.json')));
    assert(strcmp(manifest.artifactSHA256, har.hashFile(modelFile)), ...
        'Model binary was modified.');

    % モデルに添えた日本語の説明文も正しく読める
    readme = fileread(fullfile(modelFolder, 'README.md'));
    assert(contains(readme, 'モデルの開き方'));
end

fprintf('UTF8_AND_MAT_CHECK_PASSED: Japanese source/report, %d model binaries\n', ...
    numel(modelFiles));
