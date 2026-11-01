function [bundle, info] = inspectModel(id, printInfo)
% MATLABでモデルを読み込む。VSCodeでは隣のmodel-info.jsonを参照できる。
if nargin<1,id='proto_v10';end
if nargin<2,printInfo=true;end
id=char(string(id));assert(~isempty(regexp(id,'^proto_[A-Za-z0-9_]+$','once')),'Invalid model ID.');
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
folder=fullfile(root,'model',id);file=fullfile(folder,'trained.mat');
saved=load(file,'bundle');bundle=saved.bundle;
info=struct('id',bundle.spec.id,'stage',bundle.spec.stage,'algorithm',bundle.spec.algorithm, ...
    'featureSet',bundle.spec.featureSet,'featureCount',numel(bundle.featureNames), ...
    'featureNames',bundle.featureNames,'uniqueTrainingRows',numel(bundle.fitRows), ...
    'parameters',bundle.parameters,'matlabVersion',bundle.matlabVersion,'artifactSHA256',har.hashFile(file));
if isfield(bundle,'augmentedRows'),info.fittingObservations=bundle.augmentedRows;else,info.fittingObservations=numel(bundle.fitRows);end
if isfield(bundle.spec,'evaluationRole'),info.evaluationRole=bundle.spec.evaluationRole;else,info.evaluationRole='baseline_pre_tuning';end
if printInfo,disp(info);end
end
