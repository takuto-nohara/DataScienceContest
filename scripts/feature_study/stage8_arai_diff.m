function stage8_arai_diff(root, mode)
% 第8段：セミナー資料で99.9%と報告された差分24列（araiDiff24.m）を検証する。
%   mode = 'compare' : 差分24列・差分24列＋姿勢・本番の30列を、手法5つ×物差し3つで比べる
%                      （拡張なし。セミナーの分類学習器と同じ条件）
%                        random5 : 窓を無作為に5分割（分類学習器の「検証」と同じ分け方）
%                        grouped : 記録単位で5分割を3回。1回目でずれも測る
%                        test201 : 全学習データで学習し、テスト201窓で評価
%   mode = 'forward' : 本番の30列（拡張8通り＋ガウスSVM、全体で1回の標準化）に、
%                      差分の列（生の値と常用対数）を1列ずつ足して効くかを調べる
%   mode = 'confirm' : forward で上位だった列を、分割の乱数を変えた5回でもう一度確かめる
%                      （48候補から最良を選んだ偶然の当たりを除くため）
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(mode), mode = 'compare'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
g = groups(1:numel(Y));
shiftNames = ["rotate10", "rotate20", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", "noise0p01"];
shifts = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'rotate', 20, 902), ...
    perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), perturb(raw, 'noise', 0.01, 903)};
% 差分24列、本番30列（標準化前）、各軸の加速度の平均（姿勢の最も素朴な表し方）
feat = @(w) [araiDiff24(w), ergonomicFeatures(w, true), squeeze(mean(w(:, :, 1:3), 2))];
[~, araiNames] = araiDiff24(raw(1:2, :, :));
names = [araiNames, "erg" + (1:30), "acc_mean_" + ["x" "y" "z"]];
X = feat(raw);
Xte = feat(data.testRaw);
Xshift = cellfun(feat, shifts, 'UniformOutput', false);
idx = @(prefixList) find(startsWith(names, prefixList));
arai = 1:24;
erg = 25:54;
accMean = 55:57;
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'')', strrep(here, '\', '/')));

switch mode
    case 'compare'
        % 分離度（座位と立位、静止と動作）
        isSit = Y == 'Sitting'; isStand = Y == 'Standing';
        isStatic = ismember(Y, categorical(["Sitting", "Standing", "Laying"]));
        fr = @(A, a, b) (mean(A(a, :)) - mean(A(b, :))) .^ 2 ./ (var(A(a, :)) + var(A(b, :)) + eps);
        L = log10(max(X(:, arai), 1e-9));
        F = table(araiNames', fr(X(:, arai), isSit, isStand)', fr(L, isSit, isStand)', ...
            fr(X(:, arai), isStatic, ~isStatic)', median(X(isSit, arai))', median(X(isStand, arai))', ...
            median(X(Y == 'Laying', arai))', 'VariableNames', {'feature', 'fisherSitStand', ...
            'fisherSitStandLog', 'fisherStaticDynamic', 'medianSitting', 'medianStanding', 'medianLaying'});
        writetable(F, fullfile(out, 'stage8_arai_separability.csv'));
        disp(F);

        sets = struct('arai24', arai, 'arai24_accMean', [arai, accMean], 'ergo30', erg, ...
            'ergo30_arai24', [erg, arai]);
        methods = ["gauss_svm", "knn", "subspace_knn", "bagged_tree", "nnet"];
        metrics = ["random5", "grouped", "test201"];
        setNames = string(fieldnames(sets))';
        jobs = combinations(setNames, methods, metrics);
        res = cell(height(jobs), 1);
        parfor j = 1:height(jobs)
            res{j} = measure(X(:, sets.(jobs.setNames(j))), Y, Xte(:, sets.(jobs.setNames(j))), Yte, g, ...
                cellfun(@(A) A(:, sets.(jobs.setNames(j))), Xshift, 'UniformOutput', false), ...
                char(jobs.methods(j)), jobs.metrics(j));
        end
        rows = {};
        for j = 1:height(jobs)
            r = res{j};
            rows(end + 1, :) = [{jobs.setNames(j), jobs.methods(j), jobs.metrics(j), r.acc, r.sitStand}, ...
                num2cell(r.shift)]; %#ok<AGROW>
            fprintf('%-15s %-13s %-8s acc %.4f sit/stand %6.1f | %s\n', jobs.setNames(j), jobs.methods(j), ...
                jobs.metrics(j), r.acc, r.sitStand, sprintf('%.3f ', r.shift));
        end
        T = cell2table(rows, 'VariableNames', [{'featureSet', 'method', 'metric', 'accuracy', ...
            'sitStandErrors'}, cellstr("shift_" + shiftNames)]);
        writetable(T, fullfile(out, 'stage8_arai_compare.csv'));

    case {'forward', 'confirm'}
        % 本番の拡張8通りで、差分の列（生と対数）も作る
        aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
            perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
            perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
            perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
        Xaug = cellfun(feat, aug, 'UniformOutput', false);
        withLog = @(A) [A, log10(max(A(:, arai), 1e-9))];
        X = withLog(X); Xte = withLog(Xte); %#ok<NASGU>
        Xaug = cellfun(withLog, Xaug, 'UniformOutput', false);
        Xshift = cellfun(withLog, Xshift, 'UniformOutput', false);
        names = [names, "log_" + araiNames];
        candidates = [arai, 57 + (1:24)];
        opts = struct('BoxConstraint', 10);
        base = erg;
        col = @(s) find(names == s);
        if strcmp(mode, 'forward')
            jobs = [{base}, arrayfun(@(c) [base, c], candidates, 'UniformOutput', false)];
            labels = ["(base ergo30)", names(candidates)];
            seedBase = 20261115; repeats = 3;
        else
            jobs = {base, [base, col("dGX_std")], [base, col("ddGX_std")], [base, col("dGX_mean")], ...
                [base, col("dGX_std"), col("ddGX_std")]};
            labels = ["(base ergo30)", "dGX_std", "ddGX_std", "dGX_mean", "dGX_std+ddGX_std"];
            seedBase = 20271115; repeats = 5;
        end
        res = cell(numel(jobs), 1);
        parfor j = 1:numel(jobs)
            cols = jobs{j};
            o = opts;
            o.KernelScale = 1.4 * sqrt(numel(cols));
            res{j} = groupedAug(X(:, cols), cellfun(@(A) A(:, cols), Xaug, 'UniformOutput', false), Y, g, ...
                cellfun(@(A) A(:, cols), Xshift, 'UniformOutput', false), o, seedBase, repeats);
        end
        rows = {};
        for j = 1:numel(jobs)
            r = res{j};
            rows(end + 1, :) = [{labels(j), r.acc, r.accSd, r.objective, r.sitStand}, num2cell(r.shift)]; %#ok<AGROW>
            fprintf('%-20s cv %.4f (sd %.4f) J %.4f sit/stand %.1f\n', labels(j), r.acc, r.accSd, r.objective, r.sitStand);
        end
        T = cell2table(rows, 'VariableNames', [{'added', 'groupedCv', 'groupedCvSd', 'objective', 'sitStandErrors'}, ...
            cellstr("shift_" + shiftNames)]);
        T = sortrows(T, 'objective', 'descend');
        writetable(T, fullfile(out, sprintf('stage8_arai_%s.csv', mode)));
        disp(T(1:min(12, height(T)), :));
end
fprintf('STAGE8_DONE\n');
end


function r = measure(X, Y, Xte, Yte, g, Xshift, method, metric)
r.shift = nan(1, numel(Xshift));
switch metric
    case "random5"
        rng(1, 'twister');
        c = cvpartition(Y, 'KFold', 5);
        pred = Y;
        for k = 1:5
            rng(100 + k, 'twister');
            f = fitModel(method, X(training(c, k), :), Y(training(c, k)));
            pred(test(c, k)) = f(X(test(c, k), :));
        end
        r.acc = mean(pred == Y);
        r.sitStand = sitStand(Y, pred);
    case "grouped"
        ids = unique(g);
        [~, pos] = ismember(g, ids);
        acc = zeros(3, 1); ss = zeros(3, 1);
        shiftCorrect = zeros(1, numel(Xshift));
        for rep = 1:3
            s = RandStream('twister', 'Seed', 20261115 + rep);
            fold = mod(randperm(s, numel(ids)), 5)' + 1;
            fold = fold(pos);
            pred = Y;
            for k = 1:5
                tr = fold ~= k;
                rng(rep * 100 + k, 'twister');
                f = fitModel(method, X(tr, :), Y(tr));
                pred(~tr) = f(X(~tr, :));
                if rep == 1
                    for q = 1:numel(Xshift)
                        shiftCorrect(q) = shiftCorrect(q) + sum(f(Xshift{q}(~tr, :)) == Y(~tr));
                    end
                end
            end
            acc(rep) = mean(pred == Y);
            ss(rep) = sitStand(Y, pred);
        end
        r.acc = mean(acc);
        r.sitStand = mean(ss);
        r.shift = shiftCorrect / numel(Y);
    case "test201"
        rng(7, 'twister');
        f = fitModel(method, X, Y);
        pred = f(Xte);
        r.acc = mean(pred == Yte);
        r.sitStand = sitStand(Yte, pred);
end
end


function r = groupedAug(X, Xaug, Y, g, Xshift, opts, seedBase, repeats)
% 記録単位の5分割を repeats 回。学習側だけ拡張する。1回目でずれも測る。
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
    ss(rep) = sitStand(Y, pred);
end
r.acc = mean(acc);
r.accSd = std(acc);
r.sitStand = mean(ss);
r.shift = shiftCorrect / numel(Y);
r.objective = mean([r.acc, r.shift([1 3 4 5 6])]);
end


function n = sitStand(Y, pred)
n = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
