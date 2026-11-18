function stage14_production(root, mode)
% 第14段：本番配布の検証用データ（407窓、ラベル付き）を加えて、32列と、それより少ない列の組を比べる。
%   mode = 'holdout'    : 学習用2941窓（＋拡張8通り）だけで学習し、検証用407窓とテスト201窓で測る。
%                         検証用は、学習・テストの窓と64点を共有する記録につながる窓と、つながらない窓に分けて数える。
%                         検証用にずれ（装着角度・強さ・歩調・雑音）を加えた値も測る
%   mode = 'pooled'     : 学習用・テスト用・検証用の3549窓を合わせ、記録単位の5分割交差検証を5回。
%                         同じ分割で32列と各組を比べ、窓ごとの勝ち負けも数える
%   mode = 'redundancy' : 外した12列のそれぞれが、残した20列からどれだけ再現できるか（3549窓、ずれなし）。
%                         線形回帰と、決定木のバギングによる非線形回帰の R²（記録単位の5分割の分割外予測）、
%                         最も近い残した列（スピアマンの順位相関）、再現しきれない残りに行動の差が残るか（フィッシャー比）。
%                         比べるため、残した20列のそれぞれも、ほかの19列から再現する
% 列の組は stage13_backward.csv（学習用2941窓だけで決めた後ろ向き選択）から取る。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
ver = har.readVerification(root);
cls = categorical(data.classes);
Ytr = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
Yv = categorical(ver.labels, data.classes);
allRaw = cat(1, data.trainRaw, data.testRaw, ver.raw);
src = [ones(numel(Ytr), 1); 2 * ones(numel(Yte), 1); 3 * ones(numel(Yv), 1)];
Yall = [Ytr; Yte; Yv];
g = recordingGroups(allRaw);
linked = ismember(g, unique(g(src < 3)));
names = productionNames(data.trainRaw);

B = readtable(fullfile(out, 'stage13_backward.csv'), 'TextType', 'string');
sizes = [32 26 24 22 20 18];
sets = arrayfun(@(n) str2double(split(B.columns(B.nFeatures == n)))', sizes, 'UniformOutput', false);
kept20 = sets{sizes == 20};
dropped12 = setdiff(1:32, kept20, 'stable');
dropOrder = arrayfun(@(n) find(names == B.dropped(B.nFeatures == n)), 31:-1:20);

shiftNames = ["rotate10", "rotate20", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", "noise0p01"];
shiftFns = {@(w) perturb(w, 'rotate', 10, 901), @(w) perturb(w, 'rotate', 20, 902), ...
    @(w) perturb(w, 'gain', 0.7), @(w) perturb(w, 'gain', 1.3), ...
    @(w) perturb(w, 'tempo', 0.85), @(w) perturb(w, 'tempo', 1.15), @(w) perturb(w, 'noise', 0.01, 903)};
augFns = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
X = productionFeatures(allRaw);

switch mode
    case 'holdout'
        tr = src == 1;
        Xaug = cellfun(@(f) productionFeatures(f(data.trainRaw)), augFns, 'UniformOutput', false);
        Xvs = cellfun(@(f) productionFeatures(f(ver.raw)), shiftFns, 'UniformOutput', false);
        Xtrain = [X(tr, :); cell2mat(Xaug(:))];
        Ytrain = repmat(Ytr, numel(augFns) + 1, 1);
        vIdx = find(src == 3); tIdx = find(src == 2);
        lv = linked(vIdx);
        startPool();
        res = cell(numel(sets), 1);
        parfor j = 1:numel(sets)
            c = sets{j};
            f = fitModel('gauss_svm', Xtrain(:, c), Ytrain, ...
                struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(numel(c))));
            r = struct();
            r.pv = f(X(vIdx, c));
            r.pt = f(X(tIdx, c));
            r.shift = cellfun(@(A) mean(f(A(:, c)) == Yv), Xvs);
            res{j} = r;
        end
        rows = {};
        P = zeros(numel(Yv), numel(sets));
        for j = 1:numel(sets)
            r = res{j};
            okv = r.pv == Yv;
            P(:, j) = okv;
            ss = sum(ismember(Yv, cls(4:5)) & ismember(r.pv, cls(4:5)) & ~okv);
            rows(end + 1, :) = [{sizes(j), sum(okv), mean(okv), sum(okv(lv)), sum(lv), sum(okv(~lv)), sum(~lv), ...
                ss, sum(r.pt == Yte)}, num2cell(r.shift), {mean([mean(okv), r.shift([1 3 4 5 6])])}]; %#ok<AGROW>
            fprintf('%2d cols  verification %d/%d (%.4f)  linked %d/%d  unlinked %d/%d  sit/stand %d  test %d/201  rot20 %.4f\n', ...
                sizes(j), sum(okv), numel(okv), mean(okv), sum(okv(lv)), sum(lv), sum(okv(~lv)), sum(~lv), ss, ...
                sum(r.pt == Yte), r.shift(2));
            cm = confusionmat(Yv, r.pv, 'Order', cls);
            writetable(array2table(cm, 'VariableNames', cellstr(data.classes), 'RowNames', cellstr(data.classes)), ...
                fullfile(out, sprintf('stage14_holdout_confusion_%d.csv', sizes(j))), 'WriteRowNames', true);
        end
        T = cell2table(rows, 'VariableNames', [{'nFeatures', 'verCorrect', 'verAcc', 'verLinkedCorrect', 'verLinked', ...
            'verUnlinkedCorrect', 'verUnlinked', 'verSitStandErrors', 'testCorrect'}, cellstr("ver_" + shiftNames), {'verObjective'}]);
        writetable(T, fullfile(out, 'stage14_holdout.csv'));
        j20 = find(sizes == 20);
        fprintf('32 right & 20 wrong: %d, 20 right & 32 wrong: %d\n', sum(P(:, 1) & ~P(:, j20)), sum(~P(:, 1) & P(:, j20)));
        disp(T);

    case 'pooled'
        Xaug = cellfun(@(f) productionFeatures(f(allRaw)), augFns, 'UniformOutput', false);
        Xs = cellfun(@(f) productionFeatures(f(allRaw)), shiftFns, 'UniformOutput', false);
        startPool();
        repeats = 5;
        res = cell(numel(sets), 1);
        parfor j = 1:numel(sets)
            res{j} = pooledCV(X, Xaug, Yall, g, Xs, sets{j}, 20281115, repeats);
        end
        rows = {};
        for j = 1:numel(sets)
            r = res{j};
            d = r.correct - res{1}.correct;
            rows(end + 1, :) = [{sizes(j), mean(r.acc), std(r.acc), min(r.acc), mean(r.sitStand), ...
                mean(r.accBySrc(:, 1)), mean(r.accBySrc(:, 2)), mean(r.accBySrc(:, 3)), ...
                mean(sum(d < 0)), mean(sum(d > 0))}, num2cell(r.shift), {mean([mean(r.acc), r.shift([1 3 4 5 6])])}]; %#ok<AGROW>
            fprintf('%2d cols cv %.4f (sd %.4f, min %.4f) sit/stand %.1f  train %.4f test %.4f ver %.4f  vs32: lose %.1f win %.1f  rot20 %.4f\n', ...
                sizes(j), mean(r.acc), std(r.acc), min(r.acc), mean(r.sitStand), mean(r.accBySrc(:, 1)), ...
                mean(r.accBySrc(:, 2)), mean(r.accBySrc(:, 3)), mean(sum(d < 0)), mean(sum(d > 0)), r.shift(2));
        end
        T = cell2table(rows, 'VariableNames', [{'nFeatures', 'groupedCv', 'groupedCvSd', 'groupedCvMin', 'sitStandErrors', ...
            'accTrainPart', 'accTestPart', 'accVerificationPart', 'windowsLostVs32', 'windowsWonVs32'}, ...
            cellstr("shift_" + shiftNames), {'objective'}]);
        writetable(T, fullfile(out, 'stage14_pooled.csv'));
        disp(T);

    case 'redundancy'
        Z = (X - mean(X)) ./ std(X);
        ids = unique(g);
        [~, pos] = ismember(g, ids);
        s = RandStream('twister', 'Seed', 20281201);
        fold = mod(randperm(s, numel(ids)), 5)' + 1;
        fold = fold(pos);
        rho = corr(Z, 'Type', 'Spearman');
        targets = [dropOrder, kept20];
        startPool();
        res = cell(numel(targets), 1);
        parfor t = 1:numel(targets)
            j = targets(t);
            pred = setdiff(kept20, j, 'stable');
            res{t} = reconstruct(Z(:, pred), Z(:, j), Yall, fold);
        end
        rows = {};
        for t = 1:numel(targets)
            j = targets(t);
            pred = setdiff(kept20, j, 'stable');
            [~, k] = max(abs(rho(j, pred)));
            r = res{t};
            rows(end + 1, :) = {j, names(j), ismember(j, dropped12), r.r2lin, r.r2bag, names(pred(k)), rho(j, pred(k)), ...
                r.fisherOrig, r.fisherResid, r.pairOrig, r.pairResid, r.sitStandOrig, r.sitStandResid, ...
                r.r2bagStatic, r.r2bagDynamic}; %#ok<AGROW>
            fprintf('%2d %-22s dropped %d  R2 lin %.3f bag %.3f (static %.3f, dynamic %.3f)  nearest %-20s rho %+.2f  Fisher %.2f -> %.2f (%s -> %s)  sit/stand %.3f -> %.3f\n', ...
                j, names(j), ismember(j, dropped12), r.r2lin, r.r2bag, r.r2bagStatic, r.r2bagDynamic, names(pred(k)), ...
                rho(j, pred(k)), r.fisherOrig, r.fisherResid, r.pairOrig, r.pairResid, r.sitStandOrig, r.sitStandResid);
        end
        T = cell2table(rows, 'VariableNames', {'column', 'feature', 'dropped', 'r2Linear', 'r2Bagged', 'nearestKept', ...
            'spearman', 'bestFisherOriginal', 'bestFisherResidual', 'bestPairOriginal', 'bestPairResidual', ...
            'sitStandFisherOriginal', 'sitStandFisherResidual', 'r2BaggedStatic', 'r2BaggedDynamic'});
        writetable(T, fullfile(out, 'stage14_redundancy.csv'));
        disp(T);

    case 'shiftredundancy'
        % ずれのない3549窓で「残した20列 → 外した列」の再現式（決定木のバギング）を作り、
        % ずれを加えた同じ窓に当てる。重なりがずれの下でも保たれるかを見る。
        Z0mu = mean(X); Z0sd = std(X);
        Z = (X - Z0mu) ./ Z0sd;
        kinds = {'none', @(w) w; 'rotate20', @(w) perturb(w, 'rotate', 20, 902); ...
            'gain1p3', @(w) perturb(w, 'gain', 1.3); 'tempo1p15', @(w) perturb(w, 'tempo', 1.15)};
        Zs = cell(size(kinds, 1), 1);
        for q = 1:size(kinds, 1)
            Zs{q} = (productionFeatures(kinds{q, 2}(allRaw)) - Z0mu) ./ Z0sd;
        end
        rng(20281202);
        rows = {};
        for j = dropOrder
            m = fitrensemble(Z(:, kept20), Z(:, j), 'Method', 'Bag', 'NumLearningCycles', 100, ...
                'Learners', templateTree('MinLeafSize', 5));
            r2 = zeros(1, size(kinds, 1));
            for q = 1:size(kinds, 1)
                y = Zs{q}(:, j);
                r2(q) = 1 - sum((y - predict(m, Zs{q}(:, kept20))) .^ 2) / sum((y - mean(y)) .^ 2);
            end
            rows(end + 1, :) = [{j, names(j)}, num2cell(r2)]; %#ok<AGROW>
            fprintf('%2d %-22s R2 none %.3f  rotate20 %.3f  gain1.3 %.3f  tempo1.15 %.3f\n', j, names(j), r2);
        end
        T = cell2table(rows, 'VariableNames', [{'column', 'feature'}, cellstr("r2_" + string(kinds(:, 1)'))]);
        writetable(T, fullfile(out, 'stage14_shift_redundancy.csv'));
end
fprintf('STAGE14_DONE\n');
end


function startPool()
if isempty(gcp('nocreate')), parpool('Processes', 8); end
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));
end


function names = productionNames(raw)
[~, ergo] = ergonomicFeatures(raw(1:2, :, :), true);
names = [ergo([1:7, 10:30]), "gyro_x_diff_std", "gyro_x_diff2_std", "pivot_height_mean", "sway_fmean_ap"];
end


function r = pooledCV(X, Xaug, Y, g, Xs, cols, seedBase, repeats)
% 記録単位の5分割を repeats 回。学習側だけ拡張する。1回目でずれも測る。
ids = unique(g);
[~, pos] = ismember(g, ids);
n = numel(Y);
src = [ones(2941, 1); 2 * ones(201, 1); 3 * ones(n - 3142, 1)];
r.acc = zeros(repeats, 1); r.sitStand = zeros(repeats, 1); r.accBySrc = zeros(repeats, 3);
r.correct = false(n, repeats);
shiftCorrect = zeros(1, numel(Xs));
opts = struct('BoxConstraint', 10, 'KernelScale', 1.4 * sqrt(numel(cols)));
for rep = 1:repeats
    s = RandStream('twister', 'Seed', seedBase + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        tr = fold ~= k;
        Xtr = [X(tr, cols); cell2mat(cellfun(@(A) A(tr, cols), Xaug(:), 'UniformOutput', false))];
        Ytr = repmat(Y(tr), numel(Xaug) + 1, 1);
        f = fitModel('gauss_svm', Xtr, Ytr, opts);
        pred(~tr) = f(X(~tr, cols));
        if rep == 1
            for q = 1:numel(Xs)
                shiftCorrect(q) = shiftCorrect(q) + sum(f(Xs{q}(~tr, cols)) == Y(~tr));
            end
        end
    end
    ok = pred == Y;
    r.correct(:, rep) = ok;
    r.acc(rep) = mean(ok);
    r.accBySrc(rep, :) = [mean(ok(src == 1)), mean(ok(src == 2)), mean(ok(src == 3))];
    r.sitStand(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
r.shift = shiftCorrect / n;
end


function r = reconstruct(P, y, Y, fold)
% y を P から分割外で予測し、R² と、予測しきれない残りに行動の差が残るかを返す。
yl = zeros(size(y)); yb = zeros(size(y));
for k = 1:5
    tr = fold ~= k;
    A = [ones(sum(tr), 1), P(tr, :)];
    w = A \ y(tr);
    yl(~tr) = [ones(sum(~tr), 1), P(~tr, :)] * w;
    m = fitrensemble(P(tr, :), y(tr), 'Method', 'Bag', 'NumLearningCycles', 100, ...
        'Learners', templateTree('MinLeafSize', 5));
    yb(~tr) = predict(m, P(~tr, :));
end
r2 = @(yh, idx) 1 - sum((y(idx) - yh(idx)) .^ 2) / sum((y(idx) - mean(y(idx))) .^ 2);
everyRow = true(size(y));
static = ismember(Y, categorical(["Sitting", "Standing", "Laying"]));
r.r2lin = r2(yl, everyRow);
r.r2bag = r2(yb, everyRow);
r.r2bagStatic = r2(yb, static);
r.r2bagDynamic = r2(yb, ~static);
resid = y - yb;
[r.fisherOrig, r.pairOrig] = bestFisher(y, Y);
[r.fisherResid, r.pairResid] = bestFisher(resid, Y);
fr = @(v) (mean(v(Y == 'Sitting')) - mean(v(Y == 'Standing'))) ^ 2 / ...
    (var(v(Y == 'Sitting')) + var(v(Y == 'Standing')));
r.sitStandOrig = fr(y);
r.sitStandResid = fr(resid);
end


function [best, pair] = bestFisher(v, Y)
c = categories(Y);
best = 0; pair = "";
for a = 1:numel(c) - 1
    for b = a + 1:numel(c)
        x1 = v(Y == c{a}); x2 = v(Y == c{b});
        f = (mean(x1) - mean(x2)) ^ 2 / (var(x1) + var(x2));
        if f > best
            best = f; pair = string(c{a}) + "/" + string(c{b});
        end
    end
end
end
