function stage3_robust(root, configFile)
% 第3段：精度の安定性と、本番で起こりうるずれへの強さを比べる。
%   ・記録単位の5分割交差検証を5回くり返し、平均・標準偏差・最悪の回を記録する
%   ・評価側の窓だけに、装着角度・動きの強さ・歩調・雑音のずれを加えて測る
%   ・学習側には、拡張なし／強さのみ／強さ＋角度＋歩調の3通りを試す
% configFile は候補（特徴量の列名・手法・ハイパーパラメータ）を並べた JSON。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
if nargin < 2, configFile = fullfile(out, 'stage3_candidates.json'); end
cands = jsondecode(fileread(configFile));

data = har.loadData(root);
Y = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, data.trainRaw, data.testRaw));
g = groups(1:numel(Y));
[~, names] = featurePool(data.trainRaw(1, :, :));

shifts = {'none', 0; 'rotate', 10; 'rotate', 20; 'gain', 0.7; 'gain', 1.3; ...
    'tempo', 0.85; 'tempo', 1.15; 'noise', 0.02};
augments = {'none', 'gain', 'gain_rot_tempo'};

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'')', strrep(here, '\', '/')));

% 評価側に加えるずれは、波形ごと先に作っておく
shifted = cell(size(shifts, 1), 1);
for s = 1:size(shifts, 1)
    if strcmp(shifts{s, 1}, 'none')
        shifted{s} = data.trainRaw;
    else
        shifted{s} = perturb(data.trainRaw, shifts{s, 1}, shifts{s, 2}, 900 + s);
    end
end

jobs = {};
for c = 1:numel(cands)
    for a = 1:numel(augments)
        jobs(end + 1, :) = {c, augments{a}}; %#ok<AGROW>
    end
end

results = cell(size(jobs, 1), 1);
parfor j = 1:size(jobs, 1)
    cand = cands(jobs{j, 1});
    cols = find(ismember(names, string(cand.features)));
    opts = struct();
    if isfield(cand, 'kernelScale'), opts.KernelScale = cand.kernelScale; end
    if isfield(cand, 'boxConstraint'), opts.BoxConstraint = cand.boxConstraint; end
    results{j} = evaluateOne(data.trainRaw, shifted, Y, g, cols, cand.method, opts, ...
        jobs{j, 2}, data.testRaw, Yte);
end

rows = {};
for j = 1:size(jobs, 1)
    cand = cands(jobs{j, 1});
    r = results{j};
    rows(end + 1, :) = [{string(cand.name), string(cand.method), numel(cand.features), ...
        string(jobs{j, 2}), r.mean, r.sd, r.worst, r.test201}, num2cell(r.shift)]; %#ok<AGROW>
end
shiftNames = strings(1, size(shifts, 1));
for s = 1:size(shifts, 1)
    shiftNames(s) = "shift_" + shifts{s, 1} + "_" + strrep(string(shifts{s, 2}), '.', 'p');
end
T = cell2table(rows, 'VariableNames', [{'candidate', 'method', 'nFeatures', 'augment', ...
    'cvMean', 'cvSd', 'cvWorstRepeat', 'test201'}, cellstr(shiftNames)]);
T.shiftMean = mean(T{:, shiftNames(2:end)}, 2);
writetable(T, fullfile(out, 'stage3_robustness.csv'));
disp(T);
fprintf('STAGE3_DONE\n');
end


function r = evaluateOne(raw, shifted, Y, g, cols, method, opts, augment, testRaw, Yte)
ids = unique(g);
[~, pos] = ismember(g, ids);
repeats = 5;
accs = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(shifted));
shiftedX = cellfun(@(w) featureCols(w, cols), shifted, 'UniformOutput', false);
for rep = 1:repeats
    stream = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(stream, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        held = fold == k;
        [Xtr, Ytr] = augmentedTrain(raw(~held, :, :), Y(~held), cols, augment, rep * 10 + k);
        rng(rep * 100 + k, 'twister');
        f = fitModel(method, Xtr, Ytr, opts);
        pred(held) = f(shiftedX{1}(held, :));
        if rep == 1
            for s = 1:numel(shifted)
                shiftCorrect(s) = shiftCorrect(s) + sum(f(shiftedX{s}(held, :)) == Y(held));
            end
        end
    end
    accs(rep) = mean(pred == Y);
end
[Xall, Yall] = augmentedTrain(raw, Y, cols, augment, 77);
rng(7, 'twister');
f = fitModel(method, Xall, Yall, opts);
r.test201 = mean(f(featureCols(testRaw, cols)) == Yte);
r.mean = mean(accs);
r.sd = std(accs);
r.worst = min(accs);
r.shift = shiftCorrect / numel(Y);
end


function X = featureCols(raw, cols)
X = featurePool(raw);
X = X(:, cols);
end


function [X, Y] = augmentedTrain(raw, Y, cols, augment, seed)
switch augment
    case 'none'
        parts = {raw};
    case 'gain'
        parts = {raw, perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2)};
    case 'gain_rot_tempo'
        parts = {raw, perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
            perturb(raw, 'rotate', 10, seed), perturb(raw, 'rotate', 10, seed + 1), ...
            perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1)};
end
X = cell2mat(cellfun(@(w) featureCols(w, cols), parts(:), 'UniformOutput', false));
Y = repmat(Y, numel(parts), 1);
end
