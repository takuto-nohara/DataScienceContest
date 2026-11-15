function stage13_backward(root, mode, sizesOrSets)
% 第13段：本番の32列を、どこまで減らせるかを調べる。
%   mode = 'profile'  : 各列の行動ごとの中央値と、最もよく分ける行動の組（フィッシャー比）
%   mode = 'backward' : 32列から、外しても最も下がらない列を1つずつ外していく（後ろ向き選択）。
%                       各段は記録単位の交差検証1回（拡張8通り、ガウスSVM）で選ぶ。
%                       止める列数は sizesOrSets（既定 8）
%   mode = 'confirm'  : sizesOrSets（列番号の組の cell）を、分割の乱数を変えた5回で確かめる
% テスト201窓は使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
names = productionNames(raw);
X = productionFeatures(raw);

if strcmp(mode, 'profile')
    cls = categories(Y);
    med = zeros(32, numel(cls));
    for c = 1:numel(cls)
        med(:, c) = median(X(Y == cls{c}, :))';
    end
    pairs = nchoosek(1:numel(cls), 2);
    bestPair = strings(32, 1); bestF = zeros(32, 1); sitStandF = zeros(32, 1); staticDynF = zeros(32, 1);
    fr = @(a, b) (mean(a) - mean(b)) .^ 2 ./ (var(a) + var(b) + eps);
    isStatic = ismember(Y, categorical(["Sitting", "Standing", "Laying"]));
    for j = 1:32
        f = zeros(size(pairs, 1), 1);
        for p = 1:size(pairs, 1)
            f(p) = fr(X(Y == cls{pairs(p, 1)}, j), X(Y == cls{pairs(p, 2)}, j));
        end
        [bestF(j), k] = max(f);
        bestPair(j) = string(cls{pairs(k, 1)}) + "/" + string(cls{pairs(k, 2)});
        sitStandF(j) = fr(X(Y == 'Sitting', j), X(Y == 'Standing', j));
        staticDynF(j) = fr(X(isStatic, j), X(~isStatic, j));
    end
    T = [table((1:32)', names', bestPair, bestF, staticDynF, sitStandF, 'VariableNames', ...
        {'column', 'feature', 'bestPair', 'bestFisher', 'fisherStaticDynamic', 'fisherSitStand'}), ...
        array2table(med, 'VariableNames', "median_" + string(cls'))];
    writetable(T, fullfile(out, 'stage13_profile.csv'));
    disp(T);
    fprintf('STAGE13_DONE\n');
    return
end

g = recordingGroups(cat(1, raw, data.testRaw));
g = g(1:numel(Y));
shifts = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'rotate', 20, 902), ...
    perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), perturb(raw, 'noise', 0.01, 903)};
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
    perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
Xaug = cellfun(@productionFeatures, aug, 'UniformOutput', false);
Xshift = cellfun(@productionFeatures, shifts, 'UniformOutput', false);
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));
evalSet = @(cols, seedBase, repeats) groupedAug(X(:, cols), cellfun(@(A) A(:, cols), Xaug, 'UniformOutput', false), ...
    Y, g, cellfun(@(A) A(:, cols), Xshift, 'UniformOutput', false), ...
    struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(numel(cols))), seedBase, repeats);

switch mode
    case 'backward'
        if nargin < 3 || isempty(sizesOrSets), sizesOrSets = 8; end
        current = 1:32;
        r = evalSet(current, 20261115, 1);
        log = {32, "(all 32)", r.acc, r.objective, r.sitStand, strjoin(string(current), " ")};
        fprintf('%2d cols start cv %.4f J %.4f\n', 32, r.acc, r.objective);
        while numel(current) > sizesOrSets
            cand = arrayfun(@(c) setdiff(current, c), current, 'UniformOutput', false);
            res = cell(numel(cand), 1);
            parfor j = 1:numel(cand)
                res{j} = evalSet(cand{j}, 20261115, 1);
            end
            J = cellfun(@(q) q.objective, res);
            [~, k] = max(J);
            dropped = current(k);
            current = cand{k};
            q = res{k};
            log(end + 1, :) = {numel(current), names(dropped), q.acc, q.objective, q.sitStand, ...
                strjoin(string(current), " ")}; %#ok<AGROW>
            fprintf('%2d cols  -%-22s cv %.4f J %.4f sit/stand %.0f\n', numel(current), names(dropped), ...
                q.acc, q.objective, q.sitStand);
            T = cell2table(log, 'VariableNames', {'nFeatures', 'dropped', 'groupedCv1', 'objective', ...
                'sitStandErrors', 'columns'});
            writetable(T, fullfile(out, 'stage13_backward.csv'));
        end
    case 'confirm'
        sets = sizesOrSets;
        res = cell(numel(sets), 1);
        parfor j = 1:numel(sets)
            res{j} = evalSet(sets{j}, 20271115, 5);
        end
        rows = {};
        for j = 1:numel(sets)
            q = res{j};
            rows(end + 1, :) = [{numel(sets{j}), q.acc, q.accSd, q.objective, q.sitStand}, num2cell(q.shift), ...
                {strjoin(names(sets{j}), " ")}]; %#ok<AGROW>
            fprintf('%2d cols cv %.4f (sd %.4f) J %.4f sit/stand %.1f\n', numel(sets{j}), q.acc, q.accSd, ...
                q.objective, q.sitStand);
        end
        T = cell2table(rows, 'VariableNames', [{'nFeatures', 'groupedCv', 'groupedCvSd', 'objective', ...
            'sitStandErrors'}, cellstr("shift_" + ["rotate10", "rotate20", "gain0p7", "gain1p3", ...
            "tempo0p85", "tempo1p15", "noise0p01"]), {'features'}]);
        writetable(T, fullfile(out, 'stage13_confirm.csv'));
end
fprintf('STAGE13_DONE\n');
end


function names = productionNames(raw)
[~, ergo] = ergonomicFeatures(raw(1:2, :, :), true);
names = [ergo([1:7, 10:30]), "gyro_x_diff_std", "gyro_x_diff2_std", "pivot_height_mean", "sway_fmean_ap"];
end


function r = groupedAug(X, Xaug, Y, g, Xshift, opts, seedBase, repeats)
ids = unique(g);
[~, pos] = ismember(g, ids);
acc = zeros(repeats, 1); ss = zeros(repeats, 1);
shiftCorrect = zeros(1, numel(Xshift));
for rep = 1:repeats
    s = RandStream('twister', 'Seed', seedBase + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        tr = fold ~= k;
        Xtr = X(tr, :); Ytr = Y(tr);
        for a = 1:numel(Xaug)
            Xtr = [Xtr; Xaug{a}(tr, :)]; %#ok<AGROW>
            Ytr = [Ytr; Y(tr)]; %#ok<AGROW>
        end
        f = fitModel('gauss_svm', Xtr, Ytr, opts);
        pred(~tr) = f(X(~tr, :));
        if rep == 1
            for q = 1:numel(Xshift)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(Xshift{q}(~tr, :)) == Y(~tr));
            end
        end
    end
    acc(rep) = mean(pred == Y);
    ss(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
r.acc = mean(acc);
r.accSd = std(acc);
r.sitStand = mean(ss);
r.shift = shiftCorrect / numel(Y);
r.objective = mean([r.acc, r.shift([1 3 4 5 6])]);
end
