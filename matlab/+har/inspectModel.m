function [bundle, info] = inspectModel(id, printInfo)
% 保存済みのモデルを読み込み、概要を表示する。
%
%   id        : モデルの名前 (既定は 'proto_v10')
%   printInfo : 概要を画面に表示するかどうか (既定は true)
%   bundle    : 学習済みモデル一式
%   info      : 方式・特徴量・学習件数などの概要
%
% trained.mat はバイナリなので、テキストエディタでは隣の model-info.json を見る。

if nargin < 1
    id = 'proto_v10';
end
if nargin < 2
    printInfo = true;
end

id = char(string(id));
assert(~isempty(regexp(id, '^proto_[A-Za-z0-9_]+$', 'once')), 'Invalid model ID.');

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
modelFile = fullfile(root, 'model', id, 'trained.mat');
saved = load(modelFile, 'bundle');
bundle = saved.bundle;

info = struct( ...
    'id', bundle.spec.id, ...
    'stage', bundle.spec.stage, ...
    'algorithm', bundle.spec.algorithm, ...
    'featureSet', bundle.spec.featureSet, ...
    'featureCount', numel(bundle.featureNames), ...
    'featureNames', bundle.featureNames, ...
    'uniqueTrainingRows', numel(bundle.fitRows), ...
    'parameters', bundle.parameters, ...
    'matlabVersion', bundle.matlabVersion, ...
    'artifactSHA256', har.hashFile(modelFile));

% データ拡張をしたモデルでは、学習に使った件数が元の行数より多い。
if isfield(bundle, 'augmentedRows')
    info.fittingObservations = bundle.augmentedRows;
else
    info.fittingObservations = numel(bundle.fitRows);
end

% テストラベルを見て調整したモデルかどうか。記載がなければ調整前の基準モデル。
if isfield(bundle.spec, 'evaluationRole')
    info.evaluationRole = bundle.spec.evaluationRole;
else
    info.evaluationRole = 'baseline_pre_tuning';
end

if printInfo
    disp(info);
end
end
