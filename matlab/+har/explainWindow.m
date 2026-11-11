function explanation = explainWindow(bundle, raw)
% 1つの窓について、本番と同じ推論を途中経過つきでたどり直す。
% 正解ラベルは受け取らない。説明が答えを見て作られていないことを保証するため。
staticClasses = ["Sitting", "Standing", "Laying"];
bandFractionColumns = 27:32;

assert(strcmp(bundle.spec.algorithm, 'static_vote_svm'), ...
    'This explainer targets proto_v10.');
assert(size(raw, 1) == 1, 'Select exactly one window.');

% 特徴量を計算し、学習時の中心と尺度で標準化する
[features, featureNames] = har.features(raw, bundle.spec.featureSet);
assert(isequal(string(featureNames(:)), string(bundle.featureNames(:))));
standardized = (features - bundle.center) ./ bundle.scale;

% 主SVMの判定に、補助モデルの票を順に加える
[primary, negativeLoss] = predict(bundle.estimator.primary, standardized);
primary = string(primary);
specialists = bundle.estimator.specialists;
votes = strings(1, 1 + numel(specialists));
votes(1) = primary;
for k = 1:numel(specialists)
    votes(k + 1) = har.predict(specialists{k}, raw);
end

% 説明用にたどった結果が、本番の推論と食い違っていないことを確かめる
final = har.staticVote(primary, votes);
assert(final == har.predict(bundle, raw), ...
    'Explanation and production inference disagree.');

explanation = struct('raw', squeeze(raw), 'features', features, ...
    'featureNames', featureNames, 'standardized', standardized, ...
    'center', bundle.center, 'scale', bundle.scale, 'primary', primary, ...
    'votes', votes, 'final', final, ...
    'voteApplied', ismember(primary, staticClasses), ...
    'classes', string(bundle.estimator.primary.ClassNames), ...
    'negativeLoss', negativeLoss);
explanation.featureTable = table((1:numel(featureNames))', featureNames(:), ...
    features(:), bundle.center(:), bundle.scale(:), standardized(:), ...
    'VariableNames', ...
    {'column', 'feature', 'rawValue', 'trainingCenter', 'trainingScale', 'z'});

% 図に出すスペクトルが、特徴量の計算で使った値と一致することを確かめる
explanation = addSpectrum(explanation, raw);
assert(max(abs(explanation.bandFraction - features(bandFractionColumns))) < 1e-12);

explanation.pairTable = buildPairTable(bundle.estimator.primary, standardized);

% 代理モデルではなく、保存済みの決定木そのものの分岐をたどる
treeSpecialist = specialists{1};
[treeLeaf, treeLabel, treeTable] = traceTreePath(treeSpecialist, features);
explanation.treeLeaf = treeLeaf;
explanation.treeLabel = treeLabel;
assert(explanation.treeLabel == votes(2), ...
    'Tree path does not reproduce saved tree prediction.');
explanation.treeTable = treeTable;
end

function explanation = addSpectrum(explanation, raw)
% 各信号のパワースペクトルと、0.5〜3 Hz 帯が占める割合を追加する。
% 周波数の区切りは har.features (spectral32) と完全に同じにしてある。
windowLength = 128;
sampleRateHz = 50;
bandLowHz = 0.5;
bandHighHz = 3;
binCount = windowLength / 2;

% 平均を引いてから FFT し、0 Hz を除いた正の周波数だけを残す
power = abs(fft(raw - mean(raw, 2), [], 2)) .^ 2;
explanation.frequency = (1:binCount) * sampleRateHz / windowLength;
explanation.power = squeeze(power(:, 2:binCount + 1, :));

explanation.band = explanation.frequency >= bandLowHz & ...
    explanation.frequency <= bandHighHz;
bandPower = sum(explanation.power(explanation.band,:), 1);
totalPower = sum(explanation.power, 1);
explanation.bandFraction = bandPower ./ max(totalPower, eps);
end

function pairTable = buildPairTable(svm, standardized)
% 主SVMを構成する二者比較 (6クラスなら15組) それぞれのスコア、
% 選ばれたクラス、サポートベクトル数を表にまとめる。
learnerCount = numel(svm.BinaryLearners);
pair = strings(learnerCount, 1);
positive = strings(learnerCount, 1);
negative = strings(learnerCount, 1);
margin = zeros(learnerCount, 1);
selected = strings(learnerCount, 1);
support = zeros(learnerCount, 1);

for k = 1:learnerCount
    learner = svm.BinaryLearners{k};

    % 符号化行列の +1 が正のクラス、-1 が負のクラスを表す
    positiveClass = string(svm.ClassNames(svm.CodingMatrix(:,k) == 1));
    negativeClass = string(svm.ClassNames(svm.CodingMatrix(:,k) == -1));
    [label, score] = predict(learner, standardized);

    pair(k) = positiveClass + " / " + negativeClass;
    positive(k) = positiveClass;
    negative(k) = negativeClass;
    margin(k) = score(1, learner.ClassNames == 1);
    if label == 1
        selected(k) = positiveClass;
    else
        selected(k) = negativeClass;
    end
    support(k) = size(learner.SupportVectors, 1);
end

pairTable = table(pair, positive, negative, margin, selected, support);
end

function [leaf, label, treeTable] = traceTreePath(specialist, features)
% 補助の決定木を根から葉までたどり、通った分岐を表にする。
% しきい値は読みやすいように、標準化前の元の尺度へ戻して記録する。
treeFeatureColumns = 1:12;
tree = specialist.estimator;
treeZ = (features(treeFeatureColumns) - specialist.center) ./ specialist.scale;

% 経路の長さは窓ごとに違うので、事前割り当てせずに1行ずつ足していく
nodes = [];
featureNames = string.empty;
values = [];
cuts = [];
branches = string.empty;
node = 1;
while tree.IsBranchNode(node)
    column = find(string(tree.PredictorNames) == string(tree.CutPredictor{node}));
    cutPoint = tree.CutPoint(node);

    nodes(end+1, 1) = node; %#ok<AGROW>
    featureNames(end+1, 1) = specialist.featureNames(column); %#ok<AGROW>
    values(end+1, 1) = features(column); %#ok<AGROW>
    cuts(end+1, 1) = ...
        cutPoint * specialist.scale(column) + specialist.center(column); %#ok<AGROW>

    % 分岐そのものは標準化後の値で判定する
    if treeZ(column) < cutPoint
        side = 1;
        branches(end+1, 1) = "< : left"; %#ok<AGROW>
    else
        side = 2;
        branches(end+1, 1) = ">= : right"; %#ok<AGROW>
    end
    node = tree.Children(node, side);
end

leaf = node;
label = string(tree.NodeClass{node});
treeTable = table(nodes, featureNames, values, cuts, branches, ...
    'VariableNames', {'node', 'feature', 'rawValue', 'rawCut', 'branch'});
end
