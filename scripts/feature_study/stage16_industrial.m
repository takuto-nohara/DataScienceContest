function stage16_industrial(root, repeats, setNames, sets, outName)
% 第16段：産業の観点（列の数・センサの数・計算の重さ）で、32列より軽い組を比べる。
%   組は2系統。
%   ・後ろ向き選択（stage13_backward.csv）で残った 20・12・8・6 列
%   ・ジャイロを使わない組と、FFT（周波数分析）も使わない組
%   測るもの
%   ・学習用・テスト用・検証用の3549窓を合わせた、記録単位の5分割交差検証（repeats 回）
%     座位↔立位の誤り、装着角度20°のずれを加えた正解率
%   ・学習用2941窓（＋拡張8通り）だけで学習したモデルの、検証用407窓・テスト201窓の正解数
%   ・そのモデルのサポートベクターの数（推論の重さとメモリの目安）と、1窓あたりの推論時間
%   setNames・sets・outName を渡すと、その組で同じ測定をする（例：stage17_six）
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2, repeats = 3; end
if nargin < 5, outName = 'stage16_industrial.csv'; end
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
Yall = [Ytr; Yte; Yv];
g = recordingGroups(allRaw);
nTr = numel(Ytr); nTe = numel(Yte);

B = readtable(fullfile(out, 'stage13_backward.csv'), 'TextType', 'string');
pick = @(n) str2double(split(B.columns(B.nFeatures == n)))';
gyroCols = [12:15, 23, 26, 27, 29, 30, 31];   % ジャイロを使う列（31 は角加速度を使う）
fftCols = [21:24, 32];                        % FFT を使う列
accOnly = setdiff(1:32, gyroCols, 'stable');
if nargin < 4
    setNames = ["32列（本番）", "20列", "12列", "8列", "6列", ...
        "加速度のみ22列", "加速度のみ・FFTなし18列", "20列から加速度のみ14列", "12列から加速度のみ8列", ...
        "加速度のみ・FFTなし7列"];
    sets = {1:32, pick(20), pick(12), pick(8), pick(6), ...
        accOnly, setdiff(accOnly, fftCols, 'stable'), intersect(pick(20), accOnly, 'stable'), ...
        intersect(pick(12), accOnly, 'stable'), setdiff(intersect(pick(12), accOnly, 'stable'), fftCols, 'stable')};
end

augFns = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
X = productionFeatures(allRaw);
Xaug = cellfun(@(f) productionFeatures(f(allRaw)), augFns, 'UniformOutput', false);
Xrot = productionFeatures(perturb(allRaw, 'rotate', 20, 902));

if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'', ''%s'')', strrep(fullfile(root, 'matlab'), '\', '/'), strrep(here, '\', '/')));

res = cell(numel(sets), 1);
parfor j = 1:numel(sets)
    c = sets{j};
    r = struct();
    r.cv = pooledCV(X, Xaug, Yall, g, Xrot, c, 20291115, repeats);
    % 学習用2941窓だけで学習（拡張あり）
    tr = 1:nTr;
    Xtrain = [X(tr, c); cell2mat(cellfun(@(A) A(tr, c), Xaug(:), 'UniformOutput', false))];
    Ytrain = repmat(Yall(tr), numel(Xaug) + 1, 1);
    [mdl, mu, sigma] = fitSvm(Xtrain, Ytrain, numel(c));
    vIdx = nTr + nTe + (1:numel(Yv)); tIdx = nTr + (1:nTe);
    pv = predict(mdl, (X(vIdx, c) - mu) ./ sigma);
    pt = predict(mdl, (X(tIdx, c) - mu) ./ sigma);
    r.ver = sum(pv == Yall(vIdx));
    r.verSitStand = sum(ismember(Yall(vIdx), cls(4:5)) & ismember(pv, cls(4:5)) & pv ~= Yall(vIdx));
    r.test = sum(pt == Yall(tIdx));
    sv = cellfun(@(L) size(L.SupportVectors, 1), mdl.BinaryLearners);
    r.nSV = sum(sv);
    Zv = (X(vIdx, c) - mu) ./ sigma;
    t = tic; for k = 1:5, predict(mdl, Zv); end
    r.msPerWindow = toc(t) / 5 / numel(vIdx) * 1000;
    res{j} = r;
end

rows = {};
for j = 1:numel(sets)
    r = res{j}; c = sets{j};
    rows(end + 1, :) = {setNames(j), numel(c), numel(intersect(c, gyroCols)) > 0, numel(intersect(c, fftCols)), ...
        mean(r.cv.acc), std(r.cv.acc), mean(r.cv.sitStand), r.cv.rot20, r.ver, r.verSitStand, r.test, ...
        r.nSV, r.msPerWindow, strjoin(string(c), " ")}; %#ok<AGROW>
    fprintf('%-24s %2d cols  cv %.4f (sd %.4f)  sit/stand %.1f  rot20 %.4f  ver %d/407 (ss %d)  test %d/201  SV %d  %.4f ms\n', ...
        setNames(j), numel(c), mean(r.cv.acc), std(r.cv.acc), mean(r.cv.sitStand), r.cv.rot20, r.ver, r.verSitStand, ...
        r.test, r.nSV, r.msPerWindow);
end
T = cell2table(rows, 'VariableNames', {'set', 'nFeatures', 'usesGyro', 'nFftCols', 'groupedCv', 'groupedCvSd', ...
    'sitStandErrors', 'rotate20', 'verCorrect', 'verSitStandErrors', 'testCorrect', 'supportVectors', ...
    'msPerWindow', 'columns'});
writetable(T, fullfile(out, outName), 'Encoding', 'UTF-8');
fprintf('STAGE16_DONE\n');
end


function [mdl, mu, sigma] = fitSvm(X, Y, p)
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma < 1e-12) = 1;
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 1.4 * sqrt(p), 'BoxConstraint', 10);
mdl = fitcecoc((X - mu) ./ sigma, Y, 'Learners', t, 'Coding', 'onevsone');
end


function r = pooledCV(X, Xaug, Y, g, Xrot, cols, seedBase, repeats)
% 記録単位の5分割を repeats 回。学習側だけ拡張する。1回目で20°のずれも測る。
ids = unique(g);
[~, pos] = ismember(g, ids);
n = numel(Y);
r.acc = zeros(repeats, 1); r.sitStand = zeros(repeats, 1);
rotOk = 0;
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
            rotOk = rotOk + sum(f(Xrot(~tr, cols)) == Y(~tr));
        end
    end
    r.acc(rep) = mean(pred == Y);
    r.sitStand(rep) = sum(Y == 'Sitting' & pred == 'Standing') + sum(Y == 'Standing' & pred == 'Sitting');
end
r.rot20 = rotOk / n;
end
