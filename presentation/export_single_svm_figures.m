%% 単一SVM (model/single_svm_v1) の発表用の図を書き出す
% 書き出す図 (presentation/images/):
%   single_svm_v1_row176_map_overview.png : 32次元のSVMを平面で切った断面
%   single_svm_v1_row176_map_duel.png     : 窓176で競った2行動の境界
%   single_svm_v1_row176_map_growth.png   : 境界ができていく様子 (説明用)
%   single_svm_v1_test_confusion.png      : テストデータの混同行列
%
% 窓176は、多数決なしのモデルの多くが「階段を降りる」を「歩行」と取り違えた
% 窓である。混同行列は、分類学習器の「すべてテスト」のあとに表示できる
% テストデータの混同行列と同じものを描く。

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repositoryRoot, 'matlab'));
imageFolder = fullfile(repositoryRoot, 'presentation', 'images');

figures = har.decisionMap(repositoryRoot, 176, imageFolder, 'single_svm_v1');

data = har.loadData(repositoryRoot);
saved = load(fullfile(repositoryRoot, 'model', 'single_svm_v1', 'trainedModel.mat'));
predicted = saved.trainedModel.predictFcn(har.features(data.testRaw, 'gravity32'));

japanese = ["歩行"; "階段を昇る"; "階段を降りる"; "座る"; "立つ"; "横になる"];
toJapanese = @(labels) categorical(japanese(arrayfun( ...
    @(label) find(data.classes == label), string(labels))), japanese, japanese);

confusionFigure = figure('Name', 'テストデータの混同行列', 'Color', 'white', ...
    'Position', [60 60 1100 900]);
chart = confusionchart(confusionFigure, toJapanese(data.testLabels), ...
    toJapanese(predicted));
chart.Title = sprintf('テストデータ %d件の混同行列 (正解 %d件)', ...
    numel(data.testLabels), sum(string(predicted) == data.testLabels));
chart.XLabel = '予測したクラス';
chart.YLabel = '真のクラス';
chart.FontSize = 18;
exportgraphics(confusionFigure, ...
    fullfile(imageFolder, 'single_svm_v1_test_confusion.png'), 'Resolution', 150);
fprintf('SINGLE_SVM_FIGURES_EXPORTED\n');
