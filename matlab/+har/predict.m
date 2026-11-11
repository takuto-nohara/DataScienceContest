function labels = predict(bundle, raw)
% 保存済みのモデルで、センサ波形から行動名を予測する。
%
%   bundle : har.train が保存した学習済みモデル一式
%   raw    : N×128×6 のセンサ波形
%   labels : N×1 の行動名 (string)
%
% 正解ラベルは受け取らない。特徴量の計算、学習時の値による標準化、
% 分類器による判定の順に進む。

[x, names] = har.features(raw, bundle.spec.featureSet);
assert(isequal(string(names(:)), string(bundle.featureNames(:))), ...
    'Feature schema changed: retrain this model.');

% 標準化には、学習用の行だけから求めた平均と標準偏差を使う。
z = (x - bundle.center) ./ bundle.scale;

switch bundle.spec.algorithm
    case 'static_vote_svm'
        labels = predictWithStaticVote(bundle, x, z, names);
    case 'kmeans'
        labels = predictWithNearestCluster(bundle, z);
    otherwise
        labels = string(predict(bundle.estimator, z));
end

labels = labels(:);
assert(numel(labels) == size(raw, 1) && all(ismember(labels, bundle.classes)), ...
    'Invalid predictions.');
end


function labels = predictWithStaticVote(bundle, x, z, names)
% 主SVMの判定に、補助モデル (決定木と線形SVM) の判定を並べて多数決にかける。
% 補助モデルは32列のうち先頭12列 (平均と標準偏差) だけを使う。
specialistColumns = 1:12;

primary = string(predict(bundle.estimator.primary, z));
votes = primary;

for k = 1:numel(bundle.estimator.specialists)
    specialist = bundle.estimator.specialists{k};
    assert(isequal(string(names(specialistColumns)), ...
        string(specialist.featureNames)));

    % 補助モデルは、それぞれが学習したときの平均と標準偏差で標準化する。
    specialistZ = (x(:, specialistColumns) - specialist.center) ...
        ./ specialist.scale;
    votes(:, end + 1) = string(predict(specialist.estimator, specialistZ)); %#ok<AGROW>
end

labels = har.staticVote(primary, votes);
end


function labels = predictWithNearestCluster(bundle, z)
% 最も近いクラスタ中心を探し、そのクラスタに割り当てた行動名を返す。
distances = pdist2(z, bundle.estimator, 'squaredeuclidean');
[~, cluster] = min(distances, [], 2);
labels = bundle.clusterLabels(cluster);
end
