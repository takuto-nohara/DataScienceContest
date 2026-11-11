function figures = decisionMap(root, row, outputFolder)
% 32次元で働くSVMの判定を、2次元の図に切り出して見せる。
%
%   root         : リポジトリのフォルダ (省略時はこのファイルから求める)
%   row          : 説明するテスト窓の番号 (既定は 155)
%   outputFolder : 指定すると、図を PNG として書き出す
%   figures      : 3つの図のハンドル (overview, duel, growth)
%
% 作る図:
%   overview : 保存済みの主SVMを、データが最も広がる平面で切った断面
%   duel     : 選んだ窓で最後まで競った2行動について、判定に最も効く
%              2つの特徴量だけを動かしたときの境界
%   growth   : 2次元に落として学習し直した説明用のSVM。学習データが
%              増えるにつれて境界が整う様子を示す
%
% overview と duel は保存済みのモデルをそのまま使う。growth だけは説明用に
% 別のモデルを学習しており、提出するモデルの性能を示す図ではない。

if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(row)
    row = 155;
end

data = har.loadData(root);
protocol = har.protocol(root, data);
bundle = har.inspectModel('proto_v10', false);
assert(isequal(data.fingerprint, bundle.protocol.fingerprint), ...
    'Dataset does not match the saved model.');

% 図に使う3種類の点を、モデルが実際に見ている標準化後の値で用意する。
scene.bundle = bundle;
scene.classes = data.classes;
scene.fitZ = standardize(bundle, data.trainRaw(protocol.fitRows, :, :));
scene.fitLabels = data.trainLabels(protocol.fitRows);
scene.validationZ = standardize(bundle, data.trainRaw(protocol.validationRows, :, :));
scene.validationLabels = data.trainLabels(protocol.validationRows);
scene.windowZ = standardize(bundle, data.testRaw(row, :, :));
scene.windowRow = row;
scene.windowTruth = data.testLabels(row);
scene.windowPrediction = har.predict(bundle, data.testRaw(row, :, :));

figures.overview = drawOverview(scene);
figures.duel = drawDuel(scene);
figures.growth = drawGrowth(scene);

if nargin >= 3 && ~isempty(outputFolder)
    exportFigures(figures, outputFolder, row);
end
end


function z = standardize(bundle, raw)
% 波形を特徴量にし、モデルが学習時に決めた平均と標準偏差で標準化する。
x = har.features(raw, bundle.spec.featureSet);
z = (x - bundle.center) ./ bundle.scale;
end


function style = classStyle()
% 6つの行動の表示名と色。色覚の違いがあっても見分けやすい配色にしている。
style.japanese = ["歩行"; "階段を昇る"; "階段を降りる"; "座る"; "立つ"; "横になる"];
style.color = [ ...
    0.00 0.45 0.70; ...
    0.90 0.62 0.00; ...
    0.80 0.40 0.00; ...
    0.00 0.62 0.45; ...
    0.80 0.47 0.65; ...
    0.35 0.35 0.35];
end


function figureHandle = newSlideFigure(name)
% 16:9 のスライドにそのまま貼れる大きさの図を作る。
figureHandle = figure('Name', name, 'Color', 'white', ...
    'Position', [60 60 1600 900]);
end


function figureHandle = drawOverview(scene)
% 主SVMの判定を、学習データが最も広がる平面の上で塗り分ける。
style = classStyle();
primary = scene.bundle.estimator.primary;

% 平面は主成分分析で決める。第1・第2主成分が横軸と縦軸になる。
[coefficients, ~, ~, ~, explained, center] = pca(scene.fitZ);
basis = coefficients(:, 1:2);
toPlane = @(z) (z - center) * basis;
fromPlane = @(points) points * basis' + center;

validationPoints = toPlane(scene.validationZ);
windowPoint = toPlane(scene.windowZ);

% 平面上の各点を32次元へ戻し、保存済みのSVMにそのまま判定させる。
[gridX, gridY, gridPoints] = makeGrid(validationPoints, 220);
regionLabels = string(predict(primary, fromPlane(gridPoints)));
regionIndex = labelsToIndex(regionLabels, scene.classes);

% 点は平面へ投影しているので、背景の色と本来の判定が食い違うことがある。
% 図がどの程度モデルを表しているかを、一致率として示す。
actualPrediction = string(predict(primary, scene.validationZ));
planePrediction = string(predict(primary, fromPlane(validationPoints)));
agreement = mean(actualPrediction == planePrediction);

figureHandle = newSlideFigure('SVMの判定領域 (断面)');
ax = axes(figureHandle);
hold(ax, 'on');
drawRegions(ax, gridX, gridY, regionIndex, style);
handles = scatterByClass(ax, validationPoints, scene.validationLabels, ...
    scene.classes, style, 22);
windowHandle = plot(ax, windowPoint(1), windowPoint(2), 'p', ...
    'MarkerSize', 22, 'MarkerFaceColor', 'yellow', ...
    'MarkerEdgeColor', 'black', 'LineWidth', 1.5);

xlabel(ax, sprintf('第1主成分 (ばらつきの %.0f%%)', explained(1)));
ylabel(ax, sprintf('第2主成分 (ばらつきの %.0f%%)', explained(2)));
title(ax, '32次元のSVMを、データが最も広がる平面で切った断面', 'FontSize', 20);
subtitle(ax, sprintf(['背景の色 = この平面上でのSVMの判定 / 点 = 学習に使っていない' ...
    '検証データ735件 (色は正解)\n投影した点の判定が背景と一致する割合: %.0f%% ' ...
    '(残りは平面から離れた方向の違いで決まる)'], 100 * agreement), 'FontSize', 13);
legend(ax, [handles; windowHandle], ...
    [style.japanese; sprintf("説明する窓 %d", scene.windowRow)], ...
    'Location', 'eastoutside', 'FontSize', 13);
finishAxes(ax, gridX, gridY);
end


function figureHandle = drawDuel(scene)
% 選んだ窓で最後まで競った2行動について、境界と余白を描く。
style = classStyle();
primary = scene.bundle.estimator.primary;
classNames = string(primary.ClassNames);

% 主SVMのスコアが高い順に2つの行動を選ぶ。
[~, negativeLoss] = predict(primary, scene.windowZ);
[~, ranking] = sort(negativeLoss, 'descend');
contenders = classNames(ranking(1:2));

% その2行動を見分ける二者択一のSVMを、15個の中から探す。
[learner, positive, negative] = findBinaryLearner(primary, contenders);
score = @(z) positiveScore(learner, z);

% 判定に最も効く2つの特徴量を、スコアの傾きの大きさで選ぶ。
featureCount = numel(scene.windowZ);
step = 0.05;
gradient = zeros(1, featureCount);
for feature = 1:featureCount
    shift = zeros(1, featureCount);
    shift(feature) = step;
    gradient(feature) = (score(scene.windowZ + shift) ...
        - score(scene.windowZ - shift)) / (2 * step);
end
[~, order] = sort(abs(gradient), 'descend');
chosen = order(1:2);

% ほかの30個の値は選んだ窓のまま固定し、2つの特徴量だけを動かす。
isContender = ismember(scene.fitLabels, contenders);
contenderZ = scene.fitZ(isContender, :);
contenderLabels = scene.fitLabels(isContender);
[gridX, gridY, gridPoints] = makeGrid( ...
    [contenderZ(:, chosen); scene.windowZ(chosen)], 200);
sliceZ = repmat(scene.windowZ, size(gridPoints, 1), 1);
sliceZ(:, chosen) = gridPoints;
sliceScore = reshape(score(sliceZ), size(gridX));

% 軸は標準化前の、元の単位で表示する。
toRaw = @(z, feature) z * scene.bundle.scale(feature) + scene.bundle.center(feature);
rawX = toRaw(gridX, chosen(1));
rawY = toRaw(gridY, chosen(2));

figureHandle = newSlideFigure('2行動の境界');
ax = axes(figureHandle);
hold(ax, 'on');
contourf(ax, rawX, rawY, sliceScore, 24, 'LineStyle', 'none');
colormap(ax, divergingColormap( ...
    style.color(scene.classes == negative, :), ...
    style.color(scene.classes == positive, :)));
limit = max(abs(sliceScore(:)));
clim(ax, [-limit, limit]);
colorbarHandle = colorbar(ax);
colorbarHandle.Label.String = sprintf('SVMのスコア (正なら%s、負なら%s)', ...
    japaneseName(positive, scene.classes, style), ...
    japaneseName(negative, scene.classes, style));
colorbarHandle.Label.FontSize = 13;

contour(ax, rawX, rawY, sliceScore, [-1 1], 'k--', 'LineWidth', 1.2);
boundary = contour(ax, rawX, rawY, sliceScore, [0 0], 'k-', 'LineWidth', 2.5); %#ok<NASGU>

handles = gobjects(2, 1);
for k = 1:2
    members = contenderLabels == contenders(k);
    handles(k) = scatter(ax, ...
        toRaw(contenderZ(members, chosen(1)), chosen(1)), ...
        toRaw(contenderZ(members, chosen(2)), chosen(2)), 16, ...
        style.color(scene.classes == contenders(k), :), 'filled', ...
        'MarkerEdgeColor', 'white', 'MarkerFaceAlpha', 0.75);
end
windowHandle = plot(ax, toRaw(scene.windowZ(chosen(1)), chosen(1)), ...
    toRaw(scene.windowZ(chosen(2)), chosen(2)), 'p', 'MarkerSize', 24, ...
    'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);

featureNames = string(scene.bundle.featureNames);
xlabel(ax, featureNames(chosen(1)), 'Interpreter', 'none');
ylabel(ax, featureNames(chosen(2)), 'Interpreter', 'none');
title(ax, sprintf('窓 %d で最後まで競った2行動: %s 対 %s', scene.windowRow, ...
    japaneseName(contenders(1), scene.classes, style), ...
    japaneseName(contenders(2), scene.classes, style)), 'FontSize', 20);
subtitle(ax, sprintf(['太線 = 境界 (スコア0) / 破線 = 余白の端 (スコア±1) / ' ...
    '点 = 学習データ (この2特徴量へ投影)\nほかの30個の特徴量は窓 %d の値に固定。' ...
    '主SVMの判定: %s、多数決後の最終判定: %s、正解: %s'], scene.windowRow, ...
    japaneseName(contenders(1), scene.classes, style), ...
    japaneseName(scene.windowPrediction, scene.classes, style), ...
    japaneseName(scene.windowTruth, scene.classes, style)), 'FontSize', 13);
legend(ax, [handles; windowHandle], ...
    [japaneseName(contenders(1), scene.classes, style); ...
    japaneseName(contenders(2), scene.classes, style); ...
    sprintf("説明する窓 %d", scene.windowRow)], ...
    'Location', 'northeastoutside', 'FontSize', 13);
axis(ax, 'tight');
ax.FontSize = 13;
box(ax, 'on');
end


function figureHandle = drawGrowth(scene)
% 2次元に落として学習し直したSVMで、境界ができていく様子を見せる。
% 学習に使う点を増やしながら、同じ設定で4回学習する。
style = classStyle();
sampleCounts = [60, 300, 1200, numel(scene.fitLabels)];

[coefficients, ~, ~, ~, ~, center] = pca(scene.fitZ);
basis = coefficients(:, 1:2);
fitPoints = (scene.fitZ - center) * basis;
validationPoints = (scene.validationZ - center) * basis;
[gridX, gridY, gridPoints] = makeGrid(validationPoints, 160);

% 少ない件数の組が、多い件数の組に必ず含まれるよう、並べ替えを1度だけ決める。
rng(20261111, 'twister');
shuffled = randperm(numel(scene.fitLabels));

figureHandle = newSlideFigure('SVMが境界を作る様子 (説明用)');
layout = tiledlayout(figureHandle, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(sampleCounts)
    rows = shuffled(1:sampleCounts(k));
    learner = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 1, ...
        'BoxConstraint', 1, 'Standardize', false);
    model = fitcecoc(fitPoints(rows, :), scene.fitLabels(rows), ...
        'Learners', learner, 'Coding', 'onevsone', 'ClassNames', scene.classes);

    regionIndex = labelsToIndex(string(predict(model, gridPoints)), scene.classes);
    accuracy = mean(string(predict(model, validationPoints)) ...
        == scene.validationLabels);
    supportVectors = collectSupportVectors(model);

    ax = nexttile(layout);
    hold(ax, 'on');
    drawRegions(ax, gridX, gridY, regionIndex, style);
    scatterByClass(ax, fitPoints(rows, :), scene.fitLabels(rows), ...
        scene.classes, style, 12);
    plot(ax, supportVectors(:, 1), supportVectors(:, 2), 'o', ...
        'MarkerSize', 5, 'MarkerEdgeColor', 'black', 'LineWidth', 0.6);
    title(ax, sprintf('学習データ %d件 → 検証の正解率 %.1f%%', ...
        sampleCounts(k), 100 * accuracy), 'FontSize', 15);
    finishAxes(ax, gridX, gridY);
end
title(layout, 'SVMが境界を作る様子: 2次元に落として学習し直した説明用のモデル', ...
    'FontSize', 20, 'FontWeight', 'bold');
subtitle(layout, ['黒い輪 = サポートベクトル (境界の近くにあり、境界の形を決める点)。' ...
    '2次元では行動が重なるため正解率に上限がある。提出モデルは32次元で判定する。'], ...
    'FontSize', 13);
xlabel(layout, '第1主成分', 'FontSize', 14);
ylabel(layout, '第2主成分', 'FontSize', 14);
end


function [gridX, gridY, gridPoints] = makeGrid(points, resolution)
% 点の大部分が収まる範囲に、等間隔の格子を作る。極端な外れ値は範囲から外す。
lower = prctile(points, 0.5, 1);
upper = prctile(points, 99.5, 1);
margin = 0.08 * (upper - lower);
[gridX, gridY] = meshgrid( ...
    linspace(lower(1) - margin(1), upper(1) + margin(1), resolution), ...
    linspace(lower(2) - margin(2), upper(2) + margin(2), resolution));
gridPoints = [gridX(:), gridY(:)];
end


function index = labelsToIndex(labels, classes)
% 行動名を、classes の中での番号に直す。
[~, index] = ismember(labels, classes);
end


function drawRegions(ax, gridX, gridY, regionIndex, style)
% 判定の領域を、行動の色を薄くした色で塗り分ける。
paleColor = 0.72 + 0.28 * style.color;
regionImage = reshape(paleColor(regionIndex, :), [size(gridX), 3]);
image(ax, gridX(1, :), gridY(:, 1), regionImage);
contour(ax, gridX, gridY, reshape(regionIndex, size(gridX)), ...
    1.5:1:5.5, 'Color', [0.25 0.25 0.25], 'LineWidth', 0.8);
end


function handles = scatterByClass(ax, points, labels, classes, style, markerSize)
% 点を行動ごとの色で描き、凡例に使うハンドルを返す。
handles = gobjects(numel(classes), 1);
for k = 1:numel(classes)
    members = labels == classes(k);
    handles(k) = scatter(ax, points(members, 1), points(members, 2), ...
        markerSize, style.color(k, :), 'filled', ...
        'MarkerEdgeColor', 'white', 'LineWidth', 0.3);
end
end


function finishAxes(ax, gridX, gridY)
% 軸の範囲を格子に合わせ、上下が反転しないようにする。
axis(ax, [gridX(1, 1), gridX(1, end), gridY(1, 1), gridY(end, 1)]);
ax.YDir = 'normal';
ax.FontSize = 13;
box(ax, 'on');
end


function [learner, positive, negative] = findBinaryLearner(primary, contenders)
% 指定した2行動を見分ける二者択一のSVMと、その正負に対応する行動名を返す。
classNames = string(primary.ClassNames);
for k = 1:numel(primary.BinaryLearners)
    positive = classNames(primary.CodingMatrix(:, k) == 1);
    negative = classNames(primary.CodingMatrix(:, k) == -1);
    if all(ismember([positive, negative], contenders))
        learner = primary.BinaryLearners{k};
        return
    end
end
error('No binary learner separates %s and %s.', contenders(1), contenders(2));
end


function values = positiveScore(learner, z)
% 二者択一のSVMのスコアのうち、正のクラスに対する値を返す。
[~, scores] = predict(learner, z);
values = scores(:, learner.ClassNames == 1);
end


function supportVectors = collectSupportVectors(model)
% 15個の二者択一のSVMが使うサポートベクトルを、重複を除いて集める。
supportVectors = zeros(0, 2);
for k = 1:numel(model.BinaryLearners)
    supportVectors = [supportVectors; model.BinaryLearners{k}.SupportVectors]; %#ok<AGROW>
end
supportVectors = unique(supportVectors, 'rows');
end


function map = divergingColormap(negativeColor, positiveColor)
% 負の色から白を経て正の色へ変わる配色。スコア0が白になる。
steps = 128;
fade = linspace(0, 1, steps)';
towardWhite = negativeColor + fade .* (1 - negativeColor);
fromWhite = 1 + fade .* (positiveColor - 1);
map = [towardWhite; fromWhite];
end


function name = japaneseName(label, classes, style)
% 英語の行動名を、表示用の日本語にする。
name = style.japanese(classes == label);
end


function exportFigures(figures, outputFolder, row)
% 3つの図を、スライドに貼れる解像度の PNG として書き出す。
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
names = fieldnames(figures);
for k = 1:numel(names)
    file = fullfile(outputFolder, sprintf('row%03d_map_%s.png', row, names{k}));
    exportgraphics(figures.(names{k}), file, 'Resolution', 150);
    fprintf('DECISION_MAP_EXPORTED: %s\n', file);
end
end
