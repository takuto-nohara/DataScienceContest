function train(root, ids)
% model/proto_*/spec.json の設定どおりにモデルを学習し、保存する。
%
%   root : リポジトリのフォルダ
%   ids  : 学習するモデルの名前。省略するとすべてのモデルを学習する
%
% 学習に使うのは、protocol が決めた「学習に使う行」だけである。
% 検証用の行とテスト用のデータは、この関数のどこにも渡さない。

if nargin < 2
    ids = string.empty;
end

data = har.loadData(root);
protocol = har.protocol(root, data);

folders = listPrototypeFolders(root);
assert(~isempty(folders), 'No prototype specifications found.');
if ~isempty(ids)
    assert(all(ismember(string(ids), string({folders.name}))), ...
        'Unknown requested model ID.');
end

for k = 1:numel(folders)
    folder = fullfile(folders(k).folder, folders(k).name);
    spec = jsondecode(fileread(fullfile(folder, 'spec.json')));

    isRequested = isempty(ids) || ismember(string(spec.id), string(ids));
    if ~isRequested
        continue
    end

    assert(strcmp(spec.id, folders(k).name) && strcmp(spec.stage, 'prototype'), ...
        'Invalid prototype ID/stage.');
    assert(spec.trainFraction > 0 && spec.trainFraction <= 1, ...
        'Invalid trainFraction.');

    % 乱数の種を固定し、同じ設定なら同じモデルになるようにする。
    rng(spec.seed, 'twister');
    rows = selectFitRows(protocol.fitRows(:), data, spec.trainFraction);

    fitRaw = data.trainRaw(rows, :, :);
    fitLabels = data.trainLabels(rows);
    if isfield(spec, 'augmentationGains')
        [fitRaw, fitLabels] = har.augment(fitRaw, fitLabels, spec.augmentationGains);
    end

    % 標準化の平均と標準偏差は、学習に使う行だけから求める。
    [x, names] = har.features(fitRaw, spec.featureSet);
    center = mean(x, 1);
    scale = std(x, 0, 1);
    scale(scale < 1e-12) = 1;
    z = (x - center) ./ scale;

    fprintf('Training %s (%s, %d features, %d rows)\n', ...
        spec.id, spec.algorithm, size(x, 2), numel(rows));
    timer = tic;
    [estimator, parameters, clusterLabels] = ...
        fitEstimator(spec, z, fitLabels, data, rows, root);
    if isfield(spec, 'augmentationGains')
        parameters.augmentationGains = spec.augmentationGains;
    end
    duration = toc(timer);

    createdAt = char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    bundle = struct( ...
        'schemaVersion', 1, ...
        'spec', spec, ...
        'parameters', parameters, ...
        'estimator', estimator, ...
        'center', center, ...
        'scale', scale, ...
        'featureNames', names, ...
        'clusterLabels', clusterLabels, ...
        'augmentedRows', numel(fitLabels), ...
        'classes', data.classes, ...
        'fitRows', rows, ...
        'protocol', protocol, ...
        'trainingSeconds', duration, ...
        'matlabVersion', version, ...
        'createdAt', createdAt);
    saveBundle(folder, bundle, data.fingerprint);
    fprintf('Saved %s in %.2f seconds\n', spec.id, duration);
end
end


function folders = listPrototypeFolders(root)
% model/proto_vN のフォルダを、番号の小さい順に並べて返す。
% 後の番号のモデルが前の番号のモデルを部品として読み込むため、順序が要る。
folders = dir(fullfile(root, 'model', 'proto_*'));
folders = folders([folders.isdir]);

versionNumbers = arrayfun( ...
    @(folder) str2double(regexp(folder.name, '\d+$', 'match', 'once')), folders);
[~, order] = sort(versionNumbers);
folders = folders(order);
end


function rows = selectFitRows(rows, data, trainFraction)
% 学習に使う行を、行動ごとに同じ割合で無作為に間引く。割合が1なら全行を使う。
if trainFraction >= 1
    return
end

selected = [];
for label = data.classes'
    candidates = rows(data.trainLabels(rows) == label);
    keepCount = max(1, floor(numel(candidates) * trainFraction));
    selected = [selected; candidates(randperm(numel(candidates), keepCount))]; %#ok<AGROW>
end
rows = sort(selected);
end


function [estimator, parameters, clusterLabels] = ...
    fitEstimator(spec, z, labels, data, rows, root)
% spec.algorithm で指定された方式の分類器を学習する。
% clusterLabels は k-means のときだけ使い、それ以外では空になる。
clusterLabels = string.empty;

switch spec.algorithm
    case 'tree'
        parameters = struct('MaxNumSplits', 100, 'MinLeafSize', 5);
        estimator = compact(fitctree(z, labels, ...
            'MaxNumSplits', parameters.MaxNumSplits, ...
            'MinLeafSize', parameters.MinLeafSize));

    case {'linear_svm', 'rbf_svm', 'static_vote_svm'}
        [estimator, parameters] = fitSvm(spec, z, labels);
        if strcmp(spec.algorithm, 'static_vote_svm')
            [specialists, sourceHashes] = loadSpecialists(spec, data, rows, root);
            estimator = struct('primary', estimator, 'specialists', {specialists});
            parameters.staticSources = spec.staticSources;
            parameters.sourceArtifactSHA256 = sourceHashes;
            parameters.rule = ['If primary predicts Sitting/Standing/Laying, ' ...
                'use strict majority across primary and static specialists; ' ...
                'otherwise retain primary. Ties retain primary.'];
        end

    case 'bagged_trees'
        parameters = struct('NumLearningCycles', 100, ...
            'MinLeafSize', 3, 'MaxNumSplits', 200);
        learner = templateTree('MinLeafSize', parameters.MinLeafSize, ...
            'MaxNumSplits', parameters.MaxNumSplits);
        estimator = compact(fitcensemble(z, labels, 'Method', 'Bag', ...
            'NumLearningCycles', parameters.NumLearningCycles, ...
            'Learners', learner));

    case 'knn'
        parameters = struct('NumNeighbors', 5, 'DistanceWeight', 'squaredinverse');
        estimator = fitcknn(z, labels, ...
            'NumNeighbors', parameters.NumNeighbors, ...
            'DistanceWeight', parameters.DistanceWeight, 'Standardize', false);

    case 'kmeans'
        [estimator, parameters, clusterLabels] = fitClusters(z, labels, data.classes);

    otherwise
        error('Unknown algorithm: %s', spec.algorithm);
end
end


function [estimator, parameters] = fitSvm(spec, z, labels)
% 2つの行動を見分けるSVMを、行動のすべての組 (15組) について学習する。
% 特徴量はすでに標準化してあるので、SVM側の標準化は使わない。
kernel = 'gaussian';
if strcmp(spec.algorithm, 'linear_svm')
    kernel = 'linear';
end

kernelScale = 'auto';
if isfield(spec, 'kernelScale')
    kernelScale = spec.kernelScale;
end

boxConstraint = 1;
if isfield(spec, 'boxConstraint')
    boxConstraint = spec.boxConstraint;
end

learner = templateSVM('KernelFunction', kernel, 'KernelScale', kernelScale, ...
    'BoxConstraint', boxConstraint, 'Standardize', false);
estimator = compact(fitcecoc(z, labels, 'Learners', learner, 'Coding', 'onevsone'));
parameters = struct('kernel', kernel, 'KernelScale', kernelScale, ...
    'BoxConstraint', boxConstraint, 'Coding', 'onevsone');
end


function [specialists, sourceHashes] = loadSpecialists(spec, data, rows, root)
% 多数決に加える補助モデルを、保存済みのファイルから読み込む。
% 補助モデルが同じデータ・同じ行で学習されたものであることを確かめる。
sourceCount = numel(spec.staticSources);
specialists = cell(sourceCount, 1);
sourceHashes = cell(sourceCount, 1);

for k = 1:sourceCount
    source = spec.staticSources{k};
    assert(~isempty(regexp(source, '^proto_[A-Za-z0-9_]+$', 'once')));

    sourceFile = fullfile(root, 'model', source, 'trained.mat');
    manifest = jsondecode(fileread(fullfile(root, 'model', source, 'manifest.json')));
    sourceHashes{k} = har.hashFile(sourceFile);
    assert(strcmp(sourceHashes{k}, manifest.artifactSHA256));

    saved = load(sourceFile, 'bundle');
    specialists{k} = saved.bundle;
    assert(isequal(specialists{k}.fitRows(:), rows(:)) ...
        && isequal(specialists{k}.protocol.fingerprint, data.fingerprint));
    assert(strcmp(specialists{k}.spec.featureSet, 'stats12'), ...
        'Static specialist must use stats12 subset.');
end
end


function [centers, parameters, clusterLabels] = fitClusters(z, labels, classes)
% ラベルを使わずに24個のクラスタを作り、各クラスタに、そこへ入った学習
% データで最も多い行動名を割り当てる。
clusterCount = 24;
parameters = struct('clusters', clusterCount, 'Replicates', 5, 'MaxIter', 500, ...
    'mapping', 'training majority class per cluster');

[assignments, centers] = kmeans(z, clusterCount, ...
    'Replicates', parameters.Replicates, 'MaxIter', parameters.MaxIter, ...
    'EmptyAction', 'singleton');

clusterLabels = strings(clusterCount, 1);
for cluster = 1:clusterCount
    memberLabels = labels(assignments == cluster);
    counts = arrayfun(@(label) sum(memberLabels == label), classes);
    [~, mostFrequent] = max(counts);
    clusterLabels(cluster) = classes(mostFrequent);
end
end


function saveBundle(folder, bundle, fingerprint)
% 学習済みモデルを trained.mat に保存し、内容の要約を manifest.json に書く。
modelFile = fullfile(folder, 'trained.mat');
save(modelFile, 'bundle', '-v7');

% manifest にはテキストで読める情報だけを残し、大きな数値配列は除く。
metadata = rmfield(bundle, ...
    {'estimator', 'center', 'scale', 'clusterLabels', 'protocol'});
metadata.fingerprint = fingerprint;
metadata.protocolVersion = bundle.protocol.version;
metadata.artifactSHA256 = har.hashFile(modelFile);
har.writeJson(fullfile(folder, 'manifest.json'), metadata);
end
