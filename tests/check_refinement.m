% 改良後のモデル proto_v10 が、公表どおりの手順と結果になっているかを確かめる。
% 保存済みモデルで予測をやり直すだけで、学習はしない。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% データ、学習手順、保存済みモデルを読み込む
data = har.loadData(root);
protocol = har.protocol(root, data);
saved = load(fullfile(root, 'model/proto_v10/trained.mat'), 'bundle');
bundle = saved.bundle;

% 学習に使ったのは学習用の行だけで、水増しは振幅3通り分 (行数がちょうど3倍) である
gainCount = 3;
assert(isequal(bundle.fitRows(:), protocol.fitRows(:)) ...
    && bundle.augmentedRows == gainCount * numel(protocol.fitRows));

% テストデータを全問正解する
prediction = har.predict(bundle, data.testRaw);
assert(all(prediction == data.testLabels), 'Development target not achieved.');

% 窓の並び順を入れ替えても、それぞれの窓の予測は変わらない
rng(92);
shuffledOrder = randperm(numel(prediction));
shuffledPrediction = har.predict(bundle, data.testRaw(shuffledOrder, :, :));
assert(isequal(shuffledPrediction, prediction(shuffledOrder)), ...
    'Predictions depend on row order.');

% 1件ずつ単独で予測しても同じ結果になる
% (改良前に誤っていた5件と、多数決で予測が変わる155件目も含めて全件を調べる)
for row = 1:numel(prediction)
    assert(har.predict(bundle, data.testRaw(row, :, :)) == prediction(row));
end
originalErrorRows = [8; 10; 105; 176; 201];
assert(all(ismember(originalErrorRows, find(prediction == data.testLabels))));

% 改良前のモデル (proto_v1〜v7) のファイルは、学習し直しも変更もされていない
baselineModelCount = 7;
for k = 1:baselineModelCount
    modelFolder = fullfile(root, 'model', sprintf('proto_v%d', k));
    manifest = jsondecode(fileread(fullfile(modelFolder, 'manifest.json')));
    assert(strcmp(manifest.artifactSHA256, ...
        har.hashFile(fullfile(modelFolder, 'trained.mat'))));
end

fprintf(['REFINEMENT_CHECKS_PASSED: saved proto_v10 201/201, every row alone, ', ...
    'shuffled order, fitting-only augmentation, original artifacts unchanged\n']);
