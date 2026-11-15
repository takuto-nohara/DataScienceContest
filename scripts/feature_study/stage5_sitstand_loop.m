function stage5_sitstand_loop(root)
% 座位と立位だけを取り出し、静止中の微小な動き（躍度・スナップ・揺れの帯域など）で
% どこまで分けられるかを、候補を足して測るループで調べる。
%   1. 分離度   : 座位と立位のフィッシャー比（生の値と対数）
%   2. 段階追加 : 姿勢（重力方向）に候補の系統を1つずつ足し、記録単位の交差検証で測る
%   3. 前向き選択: 姿勢を出発点に、効いた系統・列だけを貪欲に残す
% テスト201窓は使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

data = har.loadData(root);
groups = recordingGroups(cat(1, data.trainRaw, data.testRaw));
Yall = categorical(data.trainLabels, data.classes);
rows = ismember(Yall, categorical(["Sitting", "Standing"]));
[Xall, names, family] = staticPool(data.trainRaw);
X = Xall(rows, :);
Y = removecats(Yall(rows));
g = groups(rows);
fprintf('sitting %d / standing %d / recordings %d / features %d\n', sum(Y == 'Sitting'), ...
    sum(Y == 'Standing'), numel(unique(g)), numel(names));

%% 1. 分離度（座位と立位のフィッシャー比、中央値）
isSit = Y == 'Sitting';
fisher = (mean(X(isSit, :)) - mean(X(~isSit, :))) .^ 2 ./ (var(X(isSit, :)) + var(X(~isSit, :)) + eps);
F = table(names', family', fisher', median(X(isSit, :))', median(X(~isSit, :))', ...
    'VariableNames', {'feature', 'family', 'fisherSitStand', 'medianSitting', 'medianStanding'});
F = sortrows(F, 'fisherSitStand', 'descend');
writetable(F, fullfile(out, 'stage5_sitstand_fisher.csv'));
disp(F(1:min(25, height(F)), :));

%% 2. 段階追加（姿勢＋系統1つ）
fams = unique(family, 'stable');
fams = fams(fams ~= "posture");
methods = ["gauss_svm", "knn", "bagged_tree", "lda"];
setNames = ["posture", "posture+" + fams, "all"];
setCols = cell(1, numel(setNames));
setCols{1} = find(family == "posture");
for k = 1:numel(fams)
    setCols{k + 1} = find(family == "posture" | family == fams(k));
end
setCols{end} = 1:numel(names);
jobs = combinations(1:numel(setNames), 1:numel(methods));
accJ = zeros(height(jobs), 1);
parfor j = 1:height(jobs)
    accJ(j) = groupedCV(X(:, setCols{jobs.Var1(j)}), Y, g, char(methods(jobs.Var2(j))), 3);
end
T = table(setNames(jobs.Var1)', methods(jobs.Var2)', cellfun(@numel, setCols(jobs.Var1))', ...
    accJ, (1 - accJ) * numel(Y), 'VariableNames', ...
    {'featureSet', 'method', 'nFeatures', 'groupedCv3', 'expectedErrors'});
writetable(T, fullfile(out, 'stage5_sitstand_increment.csv'));
disp(unstack(T(:, {'featureSet', 'method', 'groupedCv3'}), 'groupedCv3', 'method'));

%% 3. 前向き選択（系統単位、姿勢から出発）
flog = {};
for m = ["gauss_svm", "bagged_tree"]
    chosen = "posture";
    best = groupedCV(X(:, family == "posture"), Y, g, char(m), 3);
    flog(end + 1, :) = {m, 0, "posture", best}; %#ok<AGROW>
    remaining = fams;
    step = 0;
    while ~isempty(remaining)
        score = zeros(1, numel(remaining));
        parfor r = 1:numel(remaining)
            cols = ismember(family, [chosen, remaining(r)]);
            score(r) = groupedCV(X(:, cols), Y, g, char(m), 3);
        end
        [s, i] = max(score);
        if s <= best + 0.002, break; end
        step = step + 1;
        best = s;
        chosen(end + 1) = remaining(i); %#ok<AGROW>
        remaining(i) = [];
        flog(end + 1, :) = {m, step, chosen(end), best}; %#ok<AGROW>
        fprintf('family forward %-11s step %d +%-14s %.4f\n', m, step, chosen(end), best);
    end
end
L = cell2table(flog, 'VariableNames', {'method', 'step', 'added', 'groupedCv3'});
writetable(L, fullfile(out, 'stage5_sitstand_family_forward.csv'));

%% 4. 前向き選択（列単位、姿勢3列から出発、ガウスSVM）
chosen = find(family == "posture");
best = groupedCV(X(:, chosen), Y, g, 'gauss_svm', 3);
colLog = {0, "posture(3)", best};
for step = 1:10
    remaining = setdiff(1:numel(names), chosen);
    score = zeros(1, numel(remaining));
    parfor r = 1:numel(remaining)
        score(r) = groupedCV(X(:, [chosen, remaining(r)]), Y, g, 'gauss_svm', 3);
    end
    [s, i] = max(score);
    if s <= best + 0.001, break; end
    best = s;
    chosen(end + 1) = remaining(i); %#ok<AGROW>
    colLog(end + 1, :) = {step, names(remaining(i)), best}; %#ok<AGROW>
    fprintf('column forward step %d +%-28s %.4f\n', step, names(remaining(i)), best);
end
C = cell2table(colLog, 'VariableNames', {'step', 'added', 'groupedCv3'});
writetable(C, fullfile(out, 'stage5_sitstand_column_forward.csv'));
save(fullfile(out, 'stage5_static.mat'), 'Xall', 'names', 'family', 'rows', 'groups');
fprintf('STAGE5_DONE\n');
end
