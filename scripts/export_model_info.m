% 各モデルのフォルダに、概要の JSON と開き方の説明文 (README.md) を UTF-8 で出力する。
% バイナリのモデルファイル (trained.mat) は読むだけで、書き換えない。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% model/ 直下の proto_* フォルダを対象にする
modelFolders = dir(fullfile(root, 'model', 'proto_*'));
modelFolders = modelFolders([modelFolders.isdir]);

for k = 1:numel(modelFolders)
    modelId = modelFolders(k).name;
    modelFolder = fullfile(modelFolders(k).folder, modelId);

    % モデルの概要を JSON に書き出す
    [bundle, info] = har.inspectModel(modelId, false);
    har.writeJson(fullfile(modelFolder, 'model-info.json'), info);

    % README.md の本文を段落ごとに組み立てる
    heading = sprintf(['# %s：モデルの開き方\n\n', ...
        '`trained.mat` はMATLABのバイナリ形式です。', ...
        'テキストエディターで表示される不規則な文字は、', ...
        '文字コードの破損を示すものではありません。\n\n'], ...
        bundle.spec.id);
    summary = sprintf('方式：`%s` ／ 特徴量セット：`%s` ／ 特徴量数：%d個\n\n', ...
        bundle.spec.algorithm, bundle.spec.featureSet, numel(bundle.featureNames));
    usage = sprintf(['MATLABでは、リポジトリのフォルダで次を実行してください。', ...
        '`bundle` が学習済みモデル、`info` が概要です。\n\n', ...
        '```matlab\n', ...
        'addpath(''matlab'');\n', ...
        '[bundle, info] = har.inspectModel(''%s'');\n', ...
        '```\n\n'], ...
        bundle.spec.id);
    links = ['VSCodeでは [model-info.json](model-info.json) をUTF-8で開くと、', ...
        '方式・設定・特徴量名を確認できます。', ...
        '詳しい学習記録は [manifest.json](manifest.json)、', ...
        '分類の仕組みは [日本語レポート](../../reports/classification-criteria-ja.md) ', ...
        'を参照してください。', newline];

    writeUtf8Text(fullfile(modelFolder, 'README.md'), [heading, summary, usage, links]);
end

fprintf('MODEL_INFO_EXPORTED: %d models; MAT binaries unchanged\n', numel(modelFolders));

function writeUtf8Text(file, text)
% 文字列を UTF-8 のバイト列に変換して書き出す。
% OS の既定の文字コード (日本語 Windows では Shift_JIS 系) に左右されないようにする。
fileId = fopen(file, 'w', 'n', 'UTF-8');
assert(fileId ~= -1);
fwrite(fileId, unicode2native(text, 'UTF-8'), 'uint8');
fclose(fileId);
end
