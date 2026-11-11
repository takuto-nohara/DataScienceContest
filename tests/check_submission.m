% 提出用の前処理関数の下書きが、コンテストの規則を満たすかを確かめる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));
addpath(fullfile(root, 'submission'));

data = har.loadData(root);
pretreat = @(raw) Pretreatment_MATLABcontest_Advanced( ...
    raw(:, :, 1), raw(:, :, 2), raw(:, :, 3), raw(:, :, 4), raw(:, :, 5), raw(:, :, 6));

%% 出力は (試行数)×32 の有限な実数行列である
features = pretreat(data.testRaw);
assert(isequal(size(features), [size(data.testRaw, 1), 32]));
assert(isa(features, 'double') && isreal(features) && all(isfinite(features(:))));

%% 分類に使っている har.features の spectral32 と同じ値になる
assert(isequal(features, har.features(data.testRaw, 'spectral32')), ...
    '前処理関数の値が spectral32 と一致しない。');

%% 1試行だけ渡しても、順序を入れ替えても、各試行の値は変わらない
% まとめて計算するときと1件ずつ計算するときでは、足し算の順序の違いから
% 末尾の桁がわずかに変わることがある。その丸め誤差だけを許容する。
tolerance = 1e-10;
for row = 1:size(data.testRaw, 1)
    alone = pretreat(data.testRaw(row, :, :));
    assert(max(abs(alone - features(row, :))) < tolerance);
end
rng(5);
order = randperm(size(data.testRaw, 1));
shuffled = pretreat(data.testRaw(order, :, :));
assert(max(abs(shuffled - features(order, :)), [], 'all') < tolerance);

%% 値が変化しない入力でもエラーにならず、有限の値を返す
flat = pretreat(zeros(2, 128, 6));
assert(isequal(size(flat), [2, 32]) && all(isfinite(flat(:))));

%% 編集してよい範囲の目印が1つずつあり、その外に実行文がない
source = splitlines(string(fileread(fullfile(root, 'submission', ...
    'Pretreatment_MATLABcontest_Advanced.m'))));
first = find(strtrim(source) == "%ここから下を編集してよい");
last = find(strtrim(source) == "%ここまでを編集してよい");
assert(isscalar(first) && isscalar(last) && first < last);

outside = strtrim(source([1:first - 1, last + 1:end]));
outside = outside(outside ~= "" & ~startsWith(outside, "%"));
% 目印の外に残ってよいのは、関数の宣言 (2行に折り返している) と end だけ。
assert(numel(outside) == 3 && startsWith(outside(1), "function ActivityDataTrain") ...
    && outside(3) == "end", '目印の外に実行文がある。');

fprintf('SUBMISSION_CHECK_PASSED: 32 columns, equals spectral32, row-independent\n');
