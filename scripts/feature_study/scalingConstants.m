function [featureMean, featureStd, unscaled] = scalingConstants(trainRaw)
% 本番の34列（標準化前）を productionFeatures から計算し、学習データ＋拡張8通りでの
% 平均と標準偏差を返す。拡張は train_final.m と同じ。
augmentations = {@(w) perturb(w, 'gain', 0.8), @(w) perturb(w, 'gain', 1.2), ...
    @(w) perturb(w, 'rotate', 10, 11), @(w) perturb(w, 'rotate', 10, 12), ...
    @(w) perturb(w, 'tempo', 0.9), @(w) perturb(w, 'tempo', 1.1), ...
    @(w) perturb(w, 'noise', 0.005, 21), @(w) perturb(w, 'noise', 0.005, 22)};
windows = [{trainRaw}, cellfun(@(f) f(trainRaw), augmentations, 'UniformOutput', false)];
parts = cellfun(@productionFeatures, windows, 'UniformOutput', false);
unscaled = cat(1, parts{:});
featureMean = mean(unscaled, 1);
featureStd = std(unscaled, 0, 1);
featureStd(featureStd < 1e-12) = 1;
end

