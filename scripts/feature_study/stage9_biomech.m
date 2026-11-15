function stage9_biomech(root, mode, picks)
% 第9段：査読付き論文の生体計測の考え方（bioPool.m）で、座位と立位を見分けられるかを調べる。
%   mode = 'screen'  : 1. 座位と立位のフィッシャー比と、各姿勢の中央値
%                      2. 座位と立位だけの2クラス：本番32列に系統を1つずつ足す（拡張なし、ガウスSVM）
%                      3. 6クラス：本番32列（拡張8通り）に系統・列を1つずつ足す
%   mode = 'confirm' : picks（列名の配列の cell）を、分割の乱数を変えた5回で確かめ直す
% テスト201窓は使わない。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2 || isempty(mode), mode = 'screen'; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
g = groups(1:numel(Y));
shiftNames = ["rotate10", "rotate20", "gain0p7", "gain1p3", "tempo0p85", "tempo1p15", "noise0p01"];
shifts = {perturb(raw, 'rotate', 10, 901), perturb(raw, 'rotate', 20, 902), ...
    perturb(raw, 'gain', 0.7), perturb(raw, 'gain', 1.3), ...
    perturb(raw, 'tempo', 0.85), perturb(raw, 'tempo', 1.15), perturb(raw, 'noise', 0.01, 903)};
aug = {perturb(raw, 'gain', 0.8), perturb(raw, 'gain', 1.2), ...
    perturb(raw, 'rotate', 10, 11), perturb(raw, 'rotate', 10, 12), ...
    perturb(raw, 'tempo', 0.9), perturb(raw, 'tempo', 1.1), ...
    perturb(raw, 'noise', 0.005, 21), perturb(raw, 'noise', 0.005, 22)};
feat = @(w) [productionFeatures(w), bioPool(w)];
[~, bioNames, bioFamily] = bioPool(raw(1:2, :, :));
X = feat(raw);
Xaug = cellfun(feat, aug, 'UniformOutput', false);
Xshift = cellfun(feat, shifts, 'UniformOutput', false);
base = 1:32;
bio = 32 + (1:numel(bioNames));
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));
opts = struct('BoxConstraint', 10);

switch mode
    case 'screen'
        %% 1. 分離度
        isSit = Y == 'Sitting'; isStand = Y == 'Standing'; isLay = Y == 'Laying';
        B = X(:, bio);
        fr = (mean(B(isSit, :)) - mean(B(isStand, :))) .^ 2 ./ (var(B(isSit, :)) + var(B(isStand, :)) + eps);
        F = table(bioNames', bioFamily', fr', median(B(isLay, :))', median(B(isSit, :))', median(B(isStand, :))', ...
            'VariableNames', {'feature', 'family', 'fisherSitStand', 'medianLaying', 'medianSitting', 'medianStanding'});
        F = sortrows(F, 'fisherSitStand', 'descend');
        writetable(F, fullfile(out, 'stage9_bio_separability.csv'));
        disp(F);

        %% 2. 座位と立位だけ（2クラス、拡張なし）
        ss = isSit | isStand;
        Yss = removecats(Y(ss));
        fams = unique(bioFamily, 'stable');
        jobs = [{base}, cellfun(@(f) [base, bio(bioFamily == f)], cellstr(fams), 'UniformOutput', false), {[base, bio]}];
        labels = ["(base 32)", fams, "all bio"];
        acc2 = zeros(numel(jobs), 1);
        Xss = X(ss, :); gss = g(ss);
        parfor j = 1:numel(jobs)
            o = opts; o.KernelScale = 1.4 * sqrt(numel(jobs{j}));
            acc2(j) = groupedBinary(Xss(:, jobs{j}), Yss, gss, o);
        end
        T2 = table(labels', acc2, (1 - acc2) * sum(ss), 'VariableNames', {'added', 'sitStandCv', 'expectedErrors'});
        writetable(T2, fullfile(out, 'stage9_bio_sitstand.csv'));
        disp(T2);

        %% 3. 6クラス（拡張8通り）：系統と列を1つずつ
        jobs = [{base}, cellfun(@(f) [base, bio(bioFamily == f)], cellstr(fams), 'UniformOutput', false), ...
            arrayfun(@(c) [base, c], bio, 'UniformOutput', false)];
        labels = ["(base 32)", "family:" + fams, bioNames];
        run6(jobs, labels, 20261115, 3, 'stage9_bio_forward.csv');

    case 'confirm'
        col = @(s) 32 + find(bioNames == s);
        jobs = [{base}, cellfun(@(p) [base, arrayfun(col, string(p))], picks, 'UniformOutput', false)];
        labels = ["(base 32)", cellfun(@(p) strjoin(string(p), "+"), picks)];
        run6(jobs, labels, 20271115, 5, 'stage9_bio_confirm.csv');
end
fprintf('STAGE9_DONE\n');

    function run6(jobs, labels, seedBase, repeats, file)
        res = cell(numel(jobs), 1);
        parfor j = 1:numel(jobs)
            cols = jobs{j};
            o = opts; o.KernelScale = 1.4 * sqrt(numel(cols));
            res{j} = groupedAug(X(:, cols), cellfun(@(A) A(:, cols), Xaug, 'UniformOutput', false), Y, g, ...
                cellfun(@(A) A(:, cols), Xshift, 'UniformOutput', false), o, seedBase, repeats);
        end
        rows = {};
        for j = 1:numel(jobs)
            r = res{j};
            rows(end + 1, :) = [{labels(j), r.acc, r.accSd, r.objective, r.sitStand}, num2cell(r.shift)]; %#ok<AGROW>
            fprintf('%-28s cv %.4f (sd %.4f) J %.4f sit/stand %.1f\n', labels(j), r.acc, r.accSd, r.objective, r.sitStand);
        end
        Tt = cell2table(rows, 'VariableNames', [{'added', 'groupedCv', 'groupedCvSd', 'objective', 'sitStandErrors'}, ...
            cellstr("shift_" + shiftNames)]);
        Tt = sortrows(Tt, 'objective', 'descend');
        writetable(Tt, fullfile(out, file));
        disp(Tt(1:min(15, height(Tt)), :));
    end
end


function acc = groupedBinary(X, Y, g, opts)
ids = unique(g);
[~, pos] = ismember(g, ids);
a = zeros(3, 1);
for rep = 1:3
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        tr = fold ~= k;
        f = fitModel('gauss_svm', X(tr, :), Y(tr), opts);
        pred(~tr) = f(X(~tr, :));
    end
    a(rep) = mean(pred == Y);
end
acc = mean(a);
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
    ss(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
r.acc = mean(acc);
r.accSd = std(acc);
r.sitStand = mean(ss);
r.shift = shiftCorrect / numel(Y);
r.objective = mean([r.acc, r.shift([1 3 4 5 6])]);
end
