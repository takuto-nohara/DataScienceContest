% 提出用の前処理関数の下書きが、コンテストの規則を満たすかを確かめる。
%
% 下書きは2つある。
%   submission/                      spectral32 (proto_v8〜v10 と同じ特徴量)
%   submission/gravity32_single_svm/ gravity32 (多数決なしの単一SVM)
% 関数名が同じなので、1つずつパスに加えて確かめる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'));
data = har.loadData(root);

drafts = {
    fullfile(root, 'submission'), 'spectral32'
    fullfile(root, 'submission', 'gravity32_single_svm'), 'gravity32'
    };
for k = 1:size(drafts, 1)
    checkDraft(drafts{k, 1}, drafts{k, 2}, data);
    fprintf('SUBMISSION_CHECK_PASSED: %s (32 columns, equals %s, row-independent)\n', ...
        drafts{k, 1}, drafts{k, 2});
end


function checkDraft(folder, featureSet, data)
addpath(folder);
cleanup = onCleanup(@() rmpath(folder));
assert(strcmp(fileparts(which('Pretreatment_MATLABcontest_Advanced')), folder), ...
    'A different Pretreatment function shadows the draft under test.');

pretreat = @(raw) Pretreatment_MATLABcontest_Advanced( ...
    raw(:, :, 1), raw(:, :, 2), raw(:, :, 3), raw(:, :, 4), raw(:, :, 5), raw(:, :, 6));

% 出力は (試行数)×32 の有限な実数行列である。
features = pretreat(data.testRaw);
assert(isequal(size(features), [size(data.testRaw, 1), 32]));
assert(isa(features, 'double') && isreal(features) && all(isfinite(features(:))));

% 分類に使っている har.features と同じ値になる。
assert(isequal(features, har.features(data.testRaw, featureSet)), ...
    '前処理関数の値が %s と一致しない。', featureSet);

% 1試行だけ渡しても、順序を入れ替えても、各試行の値は変わらない。
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

% 値が変化しない入力でもエラーにならず、有限の値を返す。
flat = pretreat(zeros(2, 128, 6));
assert(isequal(size(flat), [2, 32]) && all(isfinite(flat(:))));

% 編集してよい範囲の目印が1つずつあり、その外に実行文がない。
source = splitlines(string(fileread(fullfile(folder, ...
    'Pretreatment_MATLABcontest_Advanced.m'))));
first = find(strtrim(source) == "%ここから下を編集してよい");
last = find(strtrim(source) == "%ここまでを編集してよい");
assert(isscalar(first) && isscalar(last) && first < last);

outside = strtrim(source([1:first - 1, last + 1:end]));
outside = outside(outside ~= "" & ~startsWith(outside, "%"));
% 目印の外に残ってよいのは、関数の宣言 (2行に折り返している) と end だけ。
assert(numel(outside) == 3 && startsWith(outside(1), "function ActivityDataTrain") ...
    && outside(3) == "end", '目印の外に実行文がある。');
end
