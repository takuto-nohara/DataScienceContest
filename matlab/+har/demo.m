function app = demo(root, row)
% 保存済みの proto_v10 が1つの窓を判定するまでの流れを、6つのタブで見せる
% 対話デモ。モデルは読み込んで実行するだけで、学習し直さない。
modelName = 'proto_v10';
pageCount = 6;

% 既定の窓 155 は、主SVMの判定が多数決で変わる例
defaultRow = 155;

if nargin < 1
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2
    row = defaultRow;
end

% データと保存済みモデルを読み込み、同じデータで学習したモデルかを確かめる
data = har.loadData(root);
bundle = har.inspectModel(modelName, false);
protocol = har.protocol(root, data);
assert(isequal(data.fingerprint, bundle.protocol.fingerprint), ...
    'Dataset does not match the saved model.');

% 評価タブに出す成績は窓を切り替えても変わらないので、起動時に一度だけ求める
development = har.predict(bundle, data.testRaw);
validation = har.predict(bundle, data.trainRaw(protocol.validationRows,:,:));
developmentMetric = har.metrics(data.testLabels, development, data.classes);
validationMetric = har.metrics( ...
    data.trainLabels(protocol.validationRows), validation, data.classes);

% 画面の骨組み: 見出し、ツールバー、タブ、ステータス行を縦に並べる
fig = uifigure('Name', '行動認識の処理デモ | proto_v10', ...
    'Position', [80 70 1420 900], 'Color', [0.97 0.98 1]);
mainGrid = uigridlayout(fig, [4 1]);
mainGrid.RowHeight = {44, 38, '1x', 36};
mainGrid.Padding = [16 10 16 10];
uilabel(mainGrid, 'Text', '6軸の波形から、行動名が決まるまで', ...
    'FontSize', 24, 'FontWeight', 'bold');

% ツールバー: 窓の選択と、実装コードを開くボタン
toolbar = uigridlayout(mainGrid, [1 6]);
toolbar.ColumnWidth = {85, 320, 180, 180, 180, '1x'};
toolbar.Padding = 0;
uilabel(toolbar, 'Text', '説明する窓', 'FontSize', 14);
items = compose('%03d | %s', (1:numel(data.testLabels))', data.testLabels);
picker = uidropdown(toolbar, 'Items', cellstr(items), ...
    'ItemsData', 1:numel(items), 'Value', row, ...
    'ValueChangedFcn', @(~,~) refresh());
addOpenCodeButton(toolbar, '特徴量のコードを開く', root, 'features.m');
addOpenCodeButton(toolbar, '判定のコードを開く', root, 'predict.m');
uibutton(toolbar, 'Text', '全画面をPNG保存', ...
    'ButtonPushedFcn', @(~,~) exportPages(fullfile(root, 'artifacts', 'demo')));
addOpenCodeButton(toolbar, '学習処理のコードを開く', root, 'train.m');

% タブ: 処理の順番どおりに6ページを用意する
tabs = uitabgroup(mainGrid);
titles = {'1 処理の全体像', '2 センサ波形', '3 特徴量と標準化', ...
    '4 SVMと多数決', '5 決定木の分岐', '6 評価'};
pages = gobjects(1, pageCount);
for tabIndex = 1:pageCount
    pages(tabIndex) = uitab(tabs, 'Title', titles{tabIndex});
end
status = uilabel(mainGrid, 'FontSize', 13, 'FontColor', [0.18 0.24 0.35]);

refresh();

% テストやコマンドラインから画面を操作するための入口
app = struct('Figure', fig, 'Tabs', tabs, 'Pages', pages, 'RowPicker', picker, ...
    'SelectRow', @selectRow, 'Export', @exportPages, ...
    'GetExplanation', @getExplanation);

    function explanation = getExplanation()
        % いま選ばれている窓の推論を、途中経過つきでたどり直す。
        explanation = har.explainWindow(bundle, data.testRaw(picker.Value,:,:));
    end

    function selectRow(newRow)
        % 窓を番号で切り替える (プログラムからの操作用)。
        validateattributes(newRow, {'numeric'}, ...
            {'scalar', 'integer', '>=', 1, '<=', numel(items)});
        picker.Value = newRow;
        refresh();
    end

    function refresh()
        % 選ばれた窓に合わせて、全タブとステータス行を描き直す。
        explanation = getExplanation();
        selectedRow = picker.Value;
        status.Text = sprintf(['窓 %d | 正解ラベル: %s | 主SVM: %s | ' ...
            '最終判定: %s | ラベルは表示・評価にだけ使用'], ...
            selectedRow, data.testLabels(selectedRow), ...
            explanation.primary, explanation.final);

        % 前の窓の表示が残らないよう、いったん全タブを空にする
        for page = 1:pageCount
            delete(pages(page).Children);
        end
        drawOverviewPage(pages(1), explanation);
        drawSignalPage(pages(2), explanation);
        drawFeaturePage(pages(3), explanation);
        drawDecisionPage(pages(4), explanation);
        drawTreePage(pages(5), explanation);
        drawEvaluationPage(pages(6), developmentMetric, validationMetric);
        drawnow;
    end

    function exportPages(out)
        % 全タブの画像と、その窓の表・要約を out フォルダに保存する。
        if ~isfolder(out)
            mkdir(out);
        end
        rowFile = @(suffix) ...
            fullfile(out, sprintf('row%03d_%s', picker.Value, suffix));

        % 保存のためにタブを順に切り替えるので、終わったら元のタブへ戻す
        previous = tabs.SelectedTab;
        cleanup = onCleanup(@() restoreTab(previous));
        for page = 1:pageCount
            tabs.SelectedTab = pages(page);
            drawnow;
            exportapp(fig, rowFile(sprintf('step%d.png', page)));
        end

        explanation = getExplanation();
        writetable(explanation.featureTable, rowFile('features.csv'), ...
            'Encoding', 'UTF-8');
        writetable(explanation.pairTable, rowFile('svm.csv'), 'Encoding', 'UTF-8');
        writetable(explanation.treeTable, rowFile('tree.csv'), 'Encoding', 'UTF-8');

        % どのモデル・データ・窓から作った画像かを、後から確かめられるように残す
        modelFile = fullfile(root, 'model', modelName, 'trained.mat');
        metadata = struct('model', modelName, ...
            'modelSHA256', har.hashFile(modelFile), ...
            'datasetFingerprint', data.fingerprint, ...
            'row', picker.Value, ...
            'truth', data.testLabels(picker.Value), ...
            'primary', explanation.primary, ...
            'votes', explanation.votes, ...
            'final', explanation.final, ...
            'voteApplied', explanation.voteApplied, ...
            'developmentCorrect', developmentMetric.correct, ...
            'developmentTotal', developmentMetric.total, ...
            'validationCorrect', validationMetric.correct, ...
            'validationTotal', validationMetric.total, ...
            'evaluationRole', 'development_tuned', ...
            'images', 'MATLAB exportapp; no model retraining');
        har.writeJson(rowFile('snapshot.json'), metadata);
        fprintf('DEMO_EXPORTED: row %d -> %s\n', picker.Value, out);
    end

    function restoreTab(tab)
        % 保存中に画面が閉じられていた場合は何もしない。
        if isvalid(fig)
            tabs.SelectedTab = tab;
        end
    end
end

%% タブごとの描画

function drawOverviewPage(tab, explanation)
% タブ1: 入力から最終判定までの6段階を、箱と矢印の流れ図で示す。
boxFaceColor = [0.88 0.93 1];
boxEdgeColor = [0.22 0.39 0.65];
arrowColor = [0.2 0.3 0.5];

layout = createPageGrid(tab, {78, '1x', 95});
addNote(layout, ['保存済みのモデルを実行します。入力は1件の6信号×128点。' ...
    '学習し直す処理はありません。']);

ax = uiaxes(layout);
ax.XLim = [0 10];
ax.YLim = [0 10];
axis(ax, 'off');
hold(ax, 'on');

% 箱は [左 下 幅 高さ]。上段を右へ、中段を左へ、下段を右へと折り返して並べる
boxes = {[0.5 7.2 3.6 1.7], [5.5 7.2 4 1.7], [5.5 4.3 4 1.7], ...
    [0.5 4.3 3.6 1.7], [0.5 1.4 3.6 1.7], [5.5 1.4 4 1.7]};
labels = {sprintf('入力波形\n6軸 × 128点 / 50 Hz'), ...
    sprintf('特徴量を計算\n平均・揺れ・相関・周期 → 32個'), ...
    sprintf('学習時の尺度で標準化\nz = (x - center) / scale'), ...
    sprintf('主RBF SVM\n15組の二者比較 → %s', explanation.primary), ...
    sprintf('静止姿勢の候補だけ多数決\n主SVM + 決定木 + 線形SVM'), ...
    sprintf('最終判定\n%s', explanation.final)};
for step = 1:numel(boxes)
    boxPosition = boxes{step};
    rectangle(ax, 'Position', boxPosition, 'Curvature', 0.08, ...
        'FaceColor', boxFaceColor, 'EdgeColor', boxEdgeColor, 'LineWidth', 1.5);
    text(ax, boxPosition(1) + boxPosition(3)/2, ...
        boxPosition(2) + boxPosition(4)/2, labels{step}, ...
        'HorizontalAlignment', 'center', 'FontSize', 15, 'Interpreter', 'none');
end

% 箱から次の箱へ向かう5本の矢印 (始点と、向き・長さ)
arrowStartX = [4.1 7.5 5.5 2.3 4.1];
arrowStartY = [8 7.2 5.1 4.3 2.2];
arrowDeltaX = [1.4 0 -1.4 0 1.4];
arrowDeltaY = [0 -1.2 0 -1.2 0];
quiver(ax, arrowStartX, arrowStartY, arrowDeltaX, arrowDeltaY, 0, ...
    'Color', arrowColor, 'LineWidth', 2, 'MaxHeadSize', 0.5);

addNote(layout, ['対応する実装: loadData.m → features.m → predict.m → ' ...
    'staticVote.m。動作の候補なら主SVMの判定をそのまま採用します。']);
end

function drawSignalPage(tab, explanation)
% タブ2: 入力の6信号を、加速度3軸と角速度3軸に分けて表示する。
windowLength = 128;
sampleRateHz = 50;
axesPerSensor = 3;

layout = createPageGrid(tab, {60, '1x', '1x', 55});
addNote(layout, ['横軸は時刻。端末のX・Y・Z軸をそのまま表示しています。' ...
    '加速度には重力が含まれ、方角の推定は行いません。']);

timeSeconds = (0:windowLength - 1) / sampleRateHz;
for sensor = 1:2
    ax = uiaxes(layout);
    columns = (sensor - 1) * axesPerSensor + (1:axesPerSensor);
    plot(ax, timeSeconds, explanation.raw(:,columns), 'LineWidth', 1.3);
    grid(ax, 'on');
    xlabel(ax, '時間 [s]');
    ylabel(ax, '配布データの値');
    legend(ax, {'X', 'Y', 'Z'}, 'Location', 'eastoutside');
    if sensor == 1
        title(ax, '加速度3軸: 端末の傾きと揺れ');
    else
        title(ax, '角速度3軸: 回転の向きと速さ');
    end
end

addNote(layout, ['128/50 = 2.56秒相当の窓です。先頭を0秒とすると最後の点は2.54秒。' ...
    '縦軸は元ファイルの数値を換算せず表示しています。']);
end

function drawFeaturePage(tab, explanation)
% タブ3: 32個の特徴量と、標準化後の値、周波数成分の例を表示する。
barColor = [0.2 0.45 0.75];
bandLowHz = 0.5;
bandHighHz = 3;

layout = createPageGrid(tab, {55, '1x', 240, 45});
addNote(layout, ['32個の数値が分類器への入力です。表のrawValueから学習時のcenterを引き、' ...
    'scaleで割ったzをSVMに渡します。']);
addTable(layout, explanation.featureTable, ...
    {'列', '特徴量', 'rawValue', '学習時center', '学習時scale', 'z'}, ...
    {45, 290, 120, 120, 120, 120});

bottom = uigridlayout(layout, [1 2]);
bottom.ColumnWidth = {'1x', '1x'};

% 左: 分類器に渡す標準化後の値
ax = uiaxes(bottom);
bar(ax, explanation.standardized, 'FaceColor', barColor);
grid(ax, 'on');
title(ax, '32特徴量の標準化後の値');
xlabel(ax, '特徴量の列番号');
ylabel(ax, 'z');

% 右: 加速度Xのスペクトル。破線は帯域パワー割合を求める範囲の両端
ax = uiaxes(bottom);
bar(ax, explanation.frequency, explanation.power(:,1));
xlim(ax, [0 6]);
grid(ax, 'on');
xline(ax, bandLowHz, '--');
xline(ax, bandHighHz, '--');
title(ax, sprintf('加速度XのFFT | 帯域パワー割合 = %.5f', ...
    explanation.bandFraction(1)));
xlabel(ax, '周波数 [Hz]');
ylabel(ax, '|FFT|^2');

addNote(layout, ['列27〜32: 0 Hzを除くパワー合計に対する0.5〜3 Hz帯の割合。' ...
    '値が大きいことだけで歩行と決める処理ではありません。']);
end

function drawDecisionPage(tab, explanation)
% タブ4: 主SVMのスコアと、静止姿勢のときだけ行う多数決の結果を表示する。
barColor = [0.2 0.45 0.75];
votes = explanation.votes;

layout = createPageGrid(tab, {58, 240, '1x', 50});
addNote(layout, sprintf( ...
    '3つの判定: 主SVM = %s / 決定木 = %s / 線形SVM = %s | 最終 = %s', ...
    votes(1), votes(2), votes(3), explanation.final));

top = uigridlayout(layout, [1 2]);
top.ColumnWidth = {'1x', '1x'};

% 左: クラスごとの主SVMのスコア
ax = uiaxes(top);
barh(ax, explanation.negativeLoss, 'FaceColor', barColor);
ax.YTick = 1:numel(explanation.classes);
ax.YTickLabel = explanation.classes;
grid(ax, 'on');
title(ax, '主SVM: 最大のNegLossを選択');
xlabel(ax, 'NegLoss（確率ではない）');

% 右: この窓で補助モデルの票が使われたかどうかの説明
if explanation.voteApplied
    gate = '多数決を適用: 3票中2票以上。3票が別なら主SVMを維持。';
else
    gate = '主SVMが動作を予測: 補助2票は最終判定に使いません。';
end
addNote(top, sprintf(['%s\n\n6行動の組み合わせは6×5÷2 = 15組です。\n' ...
    '下表は保存SVMの実スコア。正のクラスへのスコアとサポートベクトル数を表示します。\n\n' ...
    '特徴量ごとの寄与率を表す表ではありません。'], gate));

addTable(layout, explanation.pairTable, ...
    {'比較する2行動', '正のクラス', '負のクラス', '正のクラスのスコア', ...
    '二者判定', 'SV数'}, ...
    {300, 180, 180, 140, 180, 70});
addNote(layout, ['RBF SVMはサポートベクトルとの類似度を組み合わせます。' ...
    '15スコアをECOCの損失で集計し、単純な15票の最多得票では決めません。']);
end

function drawTreePage(tab, explanation)
% タブ5: 補助の決定木が、この窓で実際に通った分岐を表示する。
layout = createPageGrid(tab, {80, '1x', 75});
addNote(layout, sprintf(['補助の決定木が実際に通った分岐です。主SVMの説明とは区別します。' ...
    '到達した葉: %d / 決定木の予測: %s'], ...
    explanation.treeLeaf, explanation.treeLabel));
addTable(layout, explanation.treeTable, ...
    {'節点', '特徴量', 'この窓の値', '元の尺度のしきい値', '進んだ枝'}, ...
    {80, 330, 180, 200, 180});
addNote(layout, ['分岐は標準化後の値で計算し、表では読みやすいように元の尺度へ戻しています。' ...
    'rawValue < rawCutなら左、それ以外は右へ進みます。']);
end

function drawEvaluationPage(tab, developmentMetric, validationMetric)
% タブ6: 保存モデルの成績と、その数字から言えること・言えないことを表示する。
layout = createPageGrid(tab, {82, '1x', 110});
addNote(layout, sprintf( ...
    '保存モデルを再評価: 開発データ %d/%d = %.2f%% | 検証 %d/%d = %.2f%%', ...
    developmentMetric.correct, developmentMetric.total, ...
    100 * developmentMetric.accuracy, ...
    validationMetric.correct, validationMetric.total, ...
    100 * validationMetric.accuracy));

sideBySide = uigridlayout(layout, [1 2]);
sideBySide.ColumnWidth = {'1x', '1x'};
drawConfusionMatrix(sideBySide, developmentMetric, ...
    '開発データ: 改良時にラベルを参照済み');
drawConfusionMatrix(sideBySide, validationMetric, '検証: 固定した735件');

addNote(layout, ['201/201は開発データに対する結果です。' ...
    '改良にラベルを使っており、未知データでの100%を示しません。' ...
    '検証も人物・記録単位の分割ではないため、未見人物への性能は確認できていません。']);
end

%% 画面部品

function addOpenCodeButton(parent, label, root, fileName)
% har パッケージ内の実装ファイルをエディタで開くボタンを追加する。
codeFile = fullfile(root, 'matlab', '+har', fileName);
uibutton(parent, 'Text', label, 'ButtonPushedFcn', @(~,~) edit(codeFile));
end

function layout = createPageGrid(parent, rowHeights)
% タブの中身を縦1列に並べる、余白をそろえたレイアウトを作る。
layout = uigridlayout(parent, [numel(rowHeights) 1]);
layout.RowHeight = rowHeights;
layout.Padding = [16 12 16 12];
layout.RowSpacing = 10;
end

function label = addNote(parent, message)
% 説明文を追加する。記号をそのまま見せたいので、書式の解釈は切っておく。
label = uilabel(parent, 'Text', message, 'WordWrap', 'on', 'FontSize', 16, ...
    'VerticalAlignment', 'center', 'Interpreter', 'none');
end

function tableHandle = addTable(parent, tableData, headers, widths)
% table の中身を、編集できない表として表示する。
values = table2cell(tableData);

% 文字列のセルは char にそろえてから uitable に渡す
for k = 1:numel(values)
    if isstring(values{k})
        values{k} = char(values{k});
    end
end
tableHandle = uitable(parent, 'Data', values, 'ColumnName', headers, ...
    'ColumnWidth', widths, 'RowName', {}, 'FontSize', 14, ...
    'ColumnEditable', false);
end

function drawConfusionMatrix(parent, metric, caption)
% 混同行列を濃淡で描き、各マスに件数を重ねて表示する。
classCount = numel(metric.classes);
confusion = metric.confusion;

ax = uiaxes(parent);
imagesc(ax, confusion);
colormap(ax, flipud(gray));
axis(ax, 'square');
ax.XTick = 1:classCount;
ax.YTick = 1:classCount;
ax.XTickLabel = metric.classes;
ax.YTickLabel = metric.classes;
ax.XTickLabelRotation = 30;
xlabel(ax, '予測');
ylabel(ax, '正解');
title(ax, caption, 'FontSize', 15);

% 件数が多い濃いマスでは、文字を白にして読めるようにする
darkThreshold = max(confusion, [], 'all') / 2;
for truthIndex = 1:classCount
    for predictedIndex = 1:classCount
        count = confusion(truthIndex,predictedIndex);
        textColor = [0 0 0];
        if count > darkThreshold
            textColor = [1 1 1];
        end
        text(ax, predictedIndex, truthIndex, string(count), ...
            'HorizontalAlignment', 'center', 'Color', textColor, 'FontSize', 15);
    end
end
end
