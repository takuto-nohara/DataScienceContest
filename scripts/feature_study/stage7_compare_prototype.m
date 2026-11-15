function stage7_compare_prototype(root)
% 第7段：旧プロトタイプ（single_svm_v1、gravity32）と本番候補（ergonomic32）を
% 同じ物差しで並べる。物差しは4つ。
%   clRandomAug   : 拡張した行まで含めて無作為に5分割（分類学習器の「検証」と同じ分け方）。
%                   同じ窓の拡張コピーと、64点が重なる隣の窓が検証側に入るので高く出る
%   randomWindow  : 元の窓を無作為に5分割し、学習側だけ拡張（旧 train_single_svm.m と同じ）
%   grouped       : 64点を共有する窓を1つの記録にまとめ、記録単位で5分割（3回）。学習側だけ拡張
%   test201       : 全学習データで学習し、テスト201窓で評価（旧モデルはこの値で設定を選んでいる）
% あわせて、記録単位の分割の1回目で、本番で起こりうるずれ（装着角度・強さ・歩調・雑音）を測る。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
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

P = struct([]);
% 旧プロトタイプ：gravity32、拡張は動きの強さ0.8/1.2倍、KS 8、C 50、標準化は組ごと
oldAug = {@(w) har.augment(w, strings(size(w, 1), 1), 0.8), @(w) har.augment(w, strings(size(w, 1), 1), 1.2)};
P(1).name = "prototype_single_svm_v1";
P(1).folder = fullfile(root, 'submission', 'gravity32_single_svm');
P(1).augment = oldAug;
P(1).kernelScale = 8; P(1).box = 50; P(1).standardize = true;
% 本番候補：ergonomic32、拡張8通り、KS 7.7、C 10、標準化は前処理関数の固定の定数
P(2).name = "final_ergonomic32_svm";
P(2).folder = fullfile(root, 'submission', 'ergonomic32_svm');
P(2).augment = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
P(2).kernelScale = 7.7; P(2).box = 10; P(2).standardize = false;

for k = 1:numel(P)
    pre = @(w) featuresWith(P(k).folder, w);
    P(k).X0 = pre(raw);
    P(k).Xaug = cellfun(@(f) pre(f(raw)), P(k).augment, 'UniformOutput', false);
    P(k).Xte = pre(data.testRaw);
    P(k).Xshift = cellfun(pre, shifts, 'UniformOutput', false);
    P(k).augment = [];
end

if isempty(gcp('nocreate')), parpool('Processes', 8); end
metrics = ["clRandomAug", "randomWindow", "grouped", "test201"];
jobs = combinations(1:numel(P), 1:numel(metrics));
res = cell(height(jobs), 1);
parfor j = 1:height(jobs)
    res{j} = measure(P(jobs.Var1(j)), metrics(jobs.Var2(j)), Y, Yte, g);
end

rows = {};
for j = 1:height(jobs)
    r = res{j};
    rows(end + 1, :) = [{P(jobs.Var1(j)).name, metrics(jobs.Var2(j)), r.acc, r.sitStand}, ...
        num2cell(r.shift)]; %#ok<AGROW>
    fprintf('%-24s %-13s acc %.4f  sit/stand %5.1f | %s\n', P(jobs.Var1(j)).name, ...
        metrics(jobs.Var2(j)), r.acc, r.sitStand, sprintf('%.3f ', r.shift));
end
T = cell2table(rows, 'VariableNames', [{'model', 'metric', 'accuracy', 'sitStandErrors'}, ...
    cellstr("shift_" + shiftNames)]);
writetable(T, fullfile(out, 'stage7_prototype_comparison.csv'));
fprintf('STAGE7_DONE\n');
end


function F = featuresWith(folder, raw)
% 2つの提出用前処理関数は同じ名前なので、使う方だけをパスに入れて呼ぶ。
addpath(folder);
cleanup = onCleanup(@() rmpath(folder));
clear('Pretreatment_MATLABcontest_Advanced');
F = Pretreatment_MATLABcontest_Advanced(raw(:, :, 1), raw(:, :, 2), raw(:, :, 3), ...
    raw(:, :, 4), raw(:, :, 5), raw(:, :, 6));
end


function r = measure(p, metric, Y, Yte, g)
fit = @(X, Yt) fitcecoc(X, Yt, 'Coding', 'onevsone', 'ClassNames', categorical(categories(Y), categories(Y)), ...
    'Learners', templateSVM('KernelFunction', 'gaussian', 'KernelScale', p.kernelScale, ...
    'BoxConstraint', p.box, 'Standardize', p.standardize));
nAug = numel(p.Xaug);
Xall = [p.X0; cat(1, p.Xaug{:})];
Yall = repmat(Y, nAug + 1, 1);
r.shift = nan(1, numel(p.Xshift));
r.sitStand = NaN;
switch metric
    case "clRandomAug"
        rng(1, 'twister');
        c = cvpartition(Yall, 'KFold', 5);
        pred = Yall;
        for k = 1:5
            m = fit(Xall(training(c, k), :), Yall(training(c, k)));
            pred(test(c, k)) = predict(m, Xall(test(c, k), :));
        end
        r.acc = mean(pred == Yall);
        r.sitStand = sitStandErrors(Yall, pred);
    case "randomWindow"
        rng(1, 'twister');
        c = cvpartition(Y, 'KFold', 5);
        pred = Y;
        for k = 1:5
            tr = training(c, k);
            m = fit(stackTrain(p, tr), repmat(Y(tr), nAug + 1, 1));
            pred(test(c, k)) = predict(m, p.X0(test(c, k), :));
        end
        r.acc = mean(pred == Y);
        r.sitStand = sitStandErrors(Y, pred);
    case "grouped"
        ids = unique(g);
        [~, pos] = ismember(g, ids);
        acc = zeros(3, 1);
        ss = zeros(3, 1);
        shiftCorrect = zeros(1, numel(p.Xshift));
        for rep = 1:3
            s = RandStream('twister', 'Seed', 20261115 + rep);
            fold = mod(randperm(s, numel(ids)), 5)' + 1;
            fold = fold(pos);
            pred = Y;
            for k = 1:5
                tr = fold ~= k;
                m = fit(stackTrain(p, tr), repmat(Y(tr), nAug + 1, 1));
                pred(~tr) = predict(m, p.X0(~tr, :));
                if rep == 1
                    for q = 1:numel(p.Xshift)
                        shiftCorrect(q) = shiftCorrect(q) + sum(predict(m, p.Xshift{q}(~tr, :)) == Y(~tr));
                    end
                end
            end
            acc(rep) = mean(pred == Y);
            ss(rep) = sitStandErrors(Y, pred);
        end
        r.acc = mean(acc);
        r.sitStand = mean(ss);
        r.shift = shiftCorrect / numel(Y);
    case "test201"
        m = fit(Xall, Yall);
        pred = predict(m, p.Xte);
        r.acc = mean(pred == Yte);
        r.sitStand = sitStandErrors(Yte, pred);
end
end


function X = stackTrain(p, tr)
X = p.X0(tr, :);
for a = 1:numel(p.Xaug)
    X = [X; p.Xaug{a}(tr, :)]; %#ok<AGROW>
end
end


function n = sitStandErrors(Y, pred)
n = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
