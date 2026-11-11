function [augmented, labels] = augment(raw, labels, gains)
% 動きの大きさだけを変えた試行を作り、学習データを増やす。
%
%   raw       : N×128×6 のセンサ波形
%   labels    : N個の行動名
%   gains     : 動きに掛ける倍率の一覧 (例: [1 0.8 1.2])
%   augmented : (N×倍率の数)×128×6 の波形。倍率ごとにN件ずつ並ぶ
%
% 加速度は、試行ごとの平均 (重力とみなす) を保ち、そこからの変動だけを
% 倍率で伸縮する。端末の傾きを変えずに、動きの強さだけが違う人を模す。

assert(isvector(gains) && all(isfinite(gains)) && all(gains > 0), ...
    'Augmentation gains must be finite and positive.');

trialCount = size(raw, 1);
originalLabels = string(labels(:));
augmented = zeros(trialCount * numel(gains), 128, 6);
labels = strings(size(augmented, 1), 1);

gravity = mean(raw(:, :, 1:3), 2);
bodyMotion = raw(:, :, 1:3) - gravity;

for k = 1:numel(gains)
    rows = (k - 1) * trialCount + (1:trialCount);
    augmented(rows, :, 1:3) = gravity + gains(k) * bodyMotion;
    augmented(rows, :, 4:6) = gains(k) * raw(:, :, 4:6);
    labels(rows) = originalLabels;
end
end
