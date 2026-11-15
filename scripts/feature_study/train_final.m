function train_final(root)
% 本番候補のモデルを、提出用の前処理関数そのものを通して学習・検証する。
%   特徴量 : submission/ergonomic32_svm の前処理関数（30列＋ゼロ埋め2列。回転の揺れ3列は対数）
%   手法   : ガウスSVM、1対1、カーネルスケール 7.7、ボックス制約 10。標準化は前処理関数の
%            固定の定数で済ませるので、学習器の標準化（2クラスの組ごと）は切る
%   学習   : 学習用2941窓＋拡張8通り（動きの強さ0.8/1.2倍、装着角度10°×2、
%            歩調0.9/1.1倍、雑音0.005×2）＝26469行
% 検証は記録単位の5分割交差検証を5回。拡張した窓は学習側だけに入れる。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here, fullfile(root, 'submission', 'ergonomic32_svm'));
outModel = fullfile(root, 'model', 'final_ergonomic32_svm');
outEval = fullfile(root, 'evaluation', 'feature_study');
if ~isfolder(outModel), mkdir(outModel); end

kernelScale = 7.7;
boxConstraint = 10;
data = har.loadData(root);
raw = data.trainRaw;
Y = categorical(data.trainLabels, data.classes);
Yte = categorical(data.testLabels, data.classes);
groups = recordingGroups(cat(1, raw, data.testRaw));
g = groups(1:numel(Y));

pre = @(w) Pretreatment_MATLABcontest_Advanced(w(:, :, 1), w(:, :, 2), w(:, :, 3), ...
    w(:, :, 4), w(:, :, 5), w(:, :, 6));
augmentations = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
X0 = pre(raw);
Xaug = cellfun(@(f) pre(f(raw)), augmentations, 'UniformOutput', false);
Xte = pre(data.testRaw);
fit = @(X, Y) fitcecoc(X, Y, 'Coding', 'onevsone', 'Learners', templateSVM( ...
    'KernelFunction', 'gaussian', 'KernelScale', kernelScale, ...
    'BoxConstraint', boxConstraint, 'Standardize', false), 'ClassNames', categorical(data.classes));

%% 記録単位の交差検証（5回）
ids = unique(g);
[~, pos] = ismember(g, ids);
repeats = 5;
acc = zeros(repeats, 1);
confusionSum = zeros(6);
for rep = 1:repeats
    s = RandStream('twister', 'Seed', 20261115 + rep);
    fold = mod(randperm(s, numel(ids)), 5)' + 1;
    fold = fold(pos);
    pred = Y;
    for k = 1:5
        tr = fold ~= k;
        Xtr = [X0(tr, :); cell2mat(cellfun(@(A) A(tr, :), Xaug(:), 'UniformOutput', false))];
        Ytr = repmat(Y(tr), numel(Xaug) + 1, 1);
        m = fit(Xtr, Ytr);
        pred(~tr) = predict(m, X0(~tr, :));
    end
    acc(rep) = mean(pred == Y);
    confusionSum = confusionSum + confusionmat(Y, pred, 'Order', categorical(data.classes));
    fprintf('repeat %d grouped CV %.4f\n', rep, acc(rep));
end
fprintf('grouped CV mean %.4f sd %.4f worst %.4f\n', mean(acc), std(acc), min(acc));

%% 全学習データで学習した最終モデル
ActivityDataTrain = [X0; cell2mat(Xaug(:))];
trainActivity = repmat(Y, numel(Xaug) + 1, 1);
finalModel = fit(ActivityDataTrain, trainActivity);
testPred = predict(finalModel, Xte);
testAcc = mean(testPred == Yte);
fprintf('test201 %.4f (%d/%d)\n', testAcc, sum(testPred == Yte), numel(Yte));

% ゼロ埋めの2列が結果に影響しないことの確認（30列だけで学習したモデルと比べる）
m30 = fitcecoc(ActivityDataTrain(:, 1:30), trainActivity, 'Coding', 'onevsone', ...
    'Learners', templateSVM('KernelFunction', 'gaussian', 'KernelScale', kernelScale, ...
    'BoxConstraint', boxConstraint, 'Standardize', false), 'ClassNames', categorical(data.classes));
samePrediction = isequal(predict(m30, Xte(:, 1:30)), testPred);
fprintf('padding columns change nothing: %d\n', samePrediction);

% 拡張の乱数を変えても判定がぶれないかの確認（装着角度と雑音の乱数だけ変える）
agreement = zeros(3, 1);
for seed = 1:3
    alt = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
        @(w) perturb(w, 'rotate', 10, 100 + seed), @(w) perturb(w, 'rotate', 10, 200 + seed), ...
        @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
        @(w) perturb(w, 'noise', 0.005, 300 + seed), @(w) perturb(w, 'noise', 0.005, 400 + seed)};
    Xalt = [X0; cell2mat(cellfun(@(f) pre(f(raw)), alt(:), 'UniformOutput', false))];
    mAlt = fit(Xalt, trainActivity);
    agreement(seed) = mean(predict(mAlt, Xte) == testPred);
    fprintf('augmentation seed %d: test201 agreement with final %.4f, accuracy %.4f\n', ...
        seed, agreement(seed), mean(predict(mAlt, Xte) == Yte));
end

trainedModel = struct('ClassificationSVM', finalModel, ...
    'predictFcn', @(X) predict(finalModel, X), ...
    'About', 'Gaussian SVM (one-vs-one, no learner standardization), KernelScale 7.7, BoxConstraint 10, ergonomic32 features (gyro std as log10, standardized in Pretreatment with fixed constants)');
save(fullfile(outModel, 'trained.mat'), 'trainedModel', '-v7.3');

C = array2table(confusionSum / repeats, 'VariableNames', cellstr(data.classes), ...
    'RowNames', cellstr(data.classes));
writetable(C, fullfile(outEval, 'final_cv_confusion_mean.csv'), 'WriteRowNames', true);
info = struct('id', 'final_ergonomic32_svm', 'features', 32, 'informativeFeatures', 30, ...
    'method', 'Gaussian SVM, one-vs-one ECOC, Standardize off (fixed constants in Pretreatment)', 'kernelScale', kernelScale, ...
    'boxConstraint', boxConstraint, 'trainingRows', size(ActivityDataTrain, 1), ...
    'augmentation', 'gain 0.8/1.2, rotate 10deg x2, tempo 0.9/1.1, noise 0.005 x2', ...
    'groupedCvMean', mean(acc), 'groupedCvSd', std(acc), 'groupedCvWorst', min(acc), ...
    'groupedCvRepeats', acc', 'test201Accuracy', testAcc, ...
    'test201Correct', sum(testPred == Yte), 'paddingNeutral', samePrediction, ...
    'augmentationSeedAgreement', agreement', ...
    'test201Missed', find(testPred ~= Yte)', ...
    'test201MissedTrue', string(Yte(testPred ~= Yte))', ...
    'test201MissedPredicted', string(testPred(testPred ~= Yte))', ...
    'note', ['test201 was never used for feature or hyperparameter selection in this study. ', ...
    'Grouped CV here uses the fixed scaling constants computed from all training windows.']);
har.writeJson(fullfile(outModel, 'model-info.json'), info);
fprintf('FINAL_DONE\n');
end
