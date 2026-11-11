% 判定の根拠を説明する機能 (har.explainWindow と har.demo) が、
% 実際の予測 (har.predict) と食い違っていないかを全テスト窓で確かめる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));

% データと保存済みモデルを読み込み、説明と比べる基準の予測を求める
data = har.loadData(root);
bundle = har.inspectModel('proto_v10', false);
expected = har.predict(bundle, data.testRaw);

% proto_v10 の構成: 32特徴量、6クラスの総当たりで15組の SVM、3モデルの多数決
featureCount = 32;
pairCount = 15;
voteCount = 3;
treeVoteIndex = 2;
tolerance = 1e-12;

%% 全テスト窓の説明が、実際の計算と一致するか
changedRows = [];
for row = 1:numel(expected)
    explanation = har.explainWindow(bundle, data.testRaw(row, :, :));

    % 最終判定が har.predict と同じで、表の行数も構成どおりである
    assert(explanation.final == expected(row) ...
        && height(explanation.featureTable) == featureCount ...
        && height(explanation.pairTable) == pairCount);

    % 主 SVM の予測は、負の損失が最大のクラスである
    [~, winner] = max(explanation.negativeLoss);
    assert(explanation.classes(winner) == explanation.primary);

    % 標準化後の値は、(特徴量 - 平均) ./ 標準偏差 と一致する
    assert(all(isfinite(explanation.standardized)) ...
        && numel(explanation.votes) == voteCount);
    assert(max(abs(explanation.standardized ...
        - (explanation.features - explanation.center) ./ explanation.scale)) ...
        < tolerance);

    % 決定木の経路が示すラベルは、多数決に出した決定木の票と同じである
    assert(explanation.treeLabel == explanation.votes(treeVoteIndex));

    % 多数決で主 SVM の予測が変わった窓を控えておく
    if explanation.primary ~= explanation.final
        changedRows(end + 1) = row; %#ok<SAGROW>
    end
end

% 多数決で予測が変わるのは155件目だけである
voteChangedRow = 155;
assert(isequal(changedRows, voteChangedRow));
assert(explanation.final == expected(end));

%% 説明画面: 窓の切り替えと PNG の保存
% 途中で失敗しても画面を閉じる
app = har.demo(root, voteChangedRow);
cleanup = onCleanup(@() delete(app.Figure));
assert(app.GetExplanation().final == "Sitting");

app.SelectRow(1);
assert(app.RowPicker.Value == 1 && app.GetExplanation().final == expected(1));

% 画面のタブ1枚につき PNG が1枚、合計6枚できる
exportFolder = fullfile(root, 'artifacts', 'demo');
exportedPngCount = 6;
app.SelectRow(voteChangedRow);
app.Export(exportFolder);
assert(numel(dir(fullfile(exportFolder, 'row155_step*.png'))) == exportedPngCount);

fprintf(['EXPLANATION_CHECK_PASSED: 201 exact traces, 15 SVM pairs, ', ...
    'saved-tree paths, row switching, six PNGs\n']);
