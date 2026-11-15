function stage2_select(root, method, maxSteps)
% 第2段：どの特徴量が必須か、何列あれば足りるか。
%   (1) 系統ごとに1つずつ抜いたときの低下（除去実験）
%   (2) 1列ずつ、最も交差検証の正解率が上がる列を足していく前向き選択
% 評価は学習用2941窓の記録単位5分割交差検証。テスト201窓は参考として並べるだけで、選択には使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(method), method = 'gauss_svm'; end
if nargin < 3 || isempty(maxSteps), maxSteps = 20; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
load(fullfile(out, 'pool.mat'), 'Xtr', 'Xte', 'Ytr', 'Yte', 'names', 'family', 'gtr');

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'')', strrep(here, '\', '/')));

%% (1) 系統の除去実験（全64列から1系統ずつ抜く）
fams = unique(family, 'stable');
[baseAcc, basePred] = groupedCV(Xtr, Ytr, gtr, method, 3);
rows = cell(numel(fams), 6);
parfor k = 1:numel(fams)
    keep = family ~= fams(k);
    [a, p] = groupedCV(Xtr(:, keep), Ytr, gtr, method, 3);
    rows(k, :) = {fams(k), sum(~keep), a, a - baseAcc, ...
        pairErr(Ytr, p, 'Sitting', 'Standing') - pairErr(Ytr, basePred, 'Sitting', 'Standing'), ...
        pairErr(Ytr, p, 'WalkingUpstairs', 'WalkingDownstairs') ...
        - pairErr(Ytr, basePred, 'WalkingUpstairs', 'WalkingDownstairs')};
end
A = cell2table(rows, 'VariableNames', {'droppedFamily', 'nDropped', 'groupedCvAccuracy', ...
    'deltaVsAll', 'deltaSitStandErrors', 'deltaUpDownErrors'});
A = sortrows(A, 'deltaVsAll');
writetable(A, fullfile(out, sprintf('stage2_family_ablation_%s.csv', method)));
fprintf('all-pool %s grouped CV %.4f\n', method, baseAcc);
disp(A);

%% (2) 前向き選択
selected = [];
remaining = 1:numel(names);
curve = {};
for step = 1:maxSteps
    scores = zeros(1, numel(remaining));
    parfor j = 1:numel(remaining)
        scores(j) = groupedCV(Xtr(:, [selected, remaining(j)]), Ytr, gtr, method, 1);
    end
    [best, j] = max(scores);
    selected = [selected, remaining(j)]; %#ok<AGROW>
    remaining(j) = [];
    % 選んだ集合を3回のくり返しで測り直す（選択時の偶然の当たりを薄める）
    [acc3, pred, per] = groupedCV(Xtr(:, selected), Ytr, gtr, method, 3);
    rng(7, 'twister');
    f = fitModel(method, Xtr(:, selected), Ytr);
    testAcc = mean(f(Xte(:, selected)) == Yte);
    fprintf('step %2d +%-28s sel %.4f  rep3 %.4f (sd %.4f)  test %.4f  sit/stand %d up/down %d\n', ...
        step, names(selected(end)), best, acc3, std(per), testAcc, ...
        pairErr(Ytr, pred, 'Sitting', 'Standing'), ...
        pairErr(Ytr, pred, 'WalkingUpstairs', 'WalkingDownstairs'));
    curve(end + 1, :) = {step, names(selected(end)), family(selected(end)), best, acc3, ...
        std(per), testAcc, pairErr(Ytr, pred, 'Sitting', 'Standing'), ...
        pairErr(Ytr, pred, 'WalkingUpstairs', 'WalkingDownstairs')}; %#ok<AGROW>
    C = cell2table(curve, 'VariableNames', {'step', 'addedFeature', 'family', ...
        'selectionCv', 'groupedCv3', 'groupedCv3Sd', 'test201', 'cvSitStandErrors', 'cvUpDownErrors'});
    writetable(C, fullfile(out, sprintf('stage2_forward_%s.csv', method)));
end
fprintf('STAGE2_DONE\n');
end


function e = pairErr(Y, P, a, b)
e = sum(Y == a & P == b) + sum(Y == b & P == a);
end
