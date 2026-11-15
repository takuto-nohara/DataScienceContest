function make_figures(root)
% レポート用の図を2枚作る。数値の元は evaluation/feature_study の CSV。
%   forward_selection.png : 列を1つずつ足したときの記録単位交差検証の正解率（3手法）
%   sit_stand_tilt.png    : 座位と立位の、端末の傾き（重力方向の角度）の分布
%   sit_stand_gyro_sway.png : 座位と立位の、鉛直軸まわりのねじれ（対数）の分布
%   sit_stand_pivot.png   : 座位と立位の、骨盤の回転の支点までの距離（bioPool）の分布
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
out = fullfile(root, 'evaluation', 'feature_study');
blue = hex('#2a78d6'); orange = hex('#eb6834'); aqua = hex('#1baf7a');
ink = hex('#0b0b0b'); muted = hex('#52514e'); grid = hex('#e4e3df');
font = 'Yu Gothic UI';

%% 前向き選択の曲線
f = figure('Visible', 'off', 'Color', hex('#fcfcfb'), 'Position', [0 0 900 520]);
ax = axes(f); hold(ax, 'on');
methods = {'gauss_svm', 'ガウスSVM', blue; 'knn', 'k-NN', orange; 'bagged_tree', 'バギング木', aqua};
for k = 1:3
    T = readtable(fullfile(out, sprintf('stage2_forward_%s.csv', methods{k, 1})));
    y = 100 * T.groupedCv3;
    plot(ax, T.step, y, '-', 'Color', methods{k, 3}, 'LineWidth', 2);
    plot(ax, T.step, y, 'o', 'MarkerSize', 5, 'MarkerFaceColor', methods{k, 3}, ...
        'MarkerEdgeColor', hex('#fcfcfb'), 'LineWidth', 1);
    text(ax, T.step(end) + 0.3, y(end), sprintf('%s %.1f%%', methods{k, 2}, y(end)), ...
        'Color', ink, 'FontName', font, 'FontSize', 10, 'VerticalAlignment', 'middle');
end
yline(ax, 96.8, ':', '全71列のSVM 96.8%', 'Color', muted, 'FontName', font, ...
    'FontSize', 9, 'LabelHorizontalAlignment', 'left');
styleAxes(ax, font, ink, muted, grid);
xlim(ax, [0.5, 22.5]); ylim(ax, [62, 100]);
xlabel(ax, '使った特徴量の数（列）'); ylabel(ax, '記録単位の交差検証の正解率（%）');
title(ax, '列を1つずつ足していくと、10〜14列でほぼ頭打ちになる', 'FontWeight', 'normal');
legend(ax, findobj(ax, 'Type', 'line', 'LineStyle', '-'), flip(methods(:, 2)), ...
    'Location', 'southeast', 'Box', 'off', 'TextColor', ink);
exportgraphics(f, fullfile(out, 'forward_selection.png'), 'Resolution', 150);
close(f);

%% 座位と立位の傾き
A = readtable(fullfile(out, 'stage2b_sitstand_posture.csv'), 'TextType', 'string');
f = figure('Visible', 'off', 'Color', hex('#fcfcfb'), 'Position', [0 0 900 480]);
ax = axes(f); hold(ax, 'on');
edges = -40:2.5:50;
histogram(ax, A.pitch_deg(A.label == "Standing"), edges, 'DisplayStyle', 'stairs', ...
    'EdgeColor', blue, 'LineWidth', 2);
histogram(ax, A.pitch_deg(A.label == "Sitting"), edges, 'DisplayStyle', 'stairs', ...
    'EdgeColor', orange, 'LineWidth', 2);
ms = median(A.pitch_deg(A.label == "Standing"));
mi = median(A.pitch_deg(A.label == "Sitting"));
xline(ax, ms, '-', sprintf('立位の中央値 %.0f°', ms), 'Color', ink, 'FontName', font, 'FontSize', 9);
xline(ax, mi, '-', sprintf('座位の中央値 %.0f°', mi), 'Color', ink, 'FontName', font, 'FontSize', 9);
styleAxes(ax, font, ink, muted, grid);
xlabel(ax, '端末の傾き atan2(重力y, 重力x)（度）'); ylabel(ax, '窓の数');
title(ax, '座ると端末が約15°傾く。ただし座位の傾きは人や座り方で大きくばらつく', ...
    'FontWeight', 'normal');
legend(ax, {'立位', '座位'}, 'Box', 'off', 'TextColor', ink, 'Location', 'northeast');
exportgraphics(f, fullfile(out, 'sit_stand_tilt.png'), 'Resolution', 150);
close(f);

%% 座位と立位の回転の揺れ（鉛直軸まわり、対数の目盛り）
f = figure('Visible', 'off', 'Color', hex('#fcfcfb'), 'Position', [0 0 900 480]);
ax = axes(f); hold(ax, 'on');
edges = -3.2:0.1:0.2;
lgs = log10(max(A.gyro_std_x, 1e-9));
histogram(ax, lgs(A.label == "Standing"), edges, 'DisplayStyle', 'stairs', ...
    'EdgeColor', blue, 'LineWidth', 2);
histogram(ax, lgs(A.label == "Sitting"), edges, 'DisplayStyle', 'stairs', ...
    'EdgeColor', orange, 'LineWidth', 2);
ms = median(A.gyro_std_x(A.label == "Standing"));
mi = median(A.gyro_std_x(A.label == "Sitting"));
xline(ax, log10(ms), '-', sprintf('立位の中央値 %.3f rad/s', ms), 'Color', ink, 'FontName', font, 'FontSize', 9);
xline(ax, log10(mi), '-', sprintf('座位の中央値 %.3f rad/s', mi), 'Color', ink, 'FontName', font, 'FontSize', 9);
styleAxes(ax, font, ink, muted, grid);
xticks(ax, -3:1:0);
xticklabels(ax, {'0.001', '0.01', '0.1', '1'});
xlabel(ax, 'ジャイロxの標準偏差（鉛直軸まわりのねじれ、rad/s、対数の目盛り）'); ylabel(ax, '窓の数');
title(ax, '立っていると骨盤のねじれが約3倍になる。差は比で現れるので対数で扱う', ...
    'FontWeight', 'normal');
legend(ax, {'立位', '座位'}, 'Box', 'off', 'TextColor', ink, 'Location', 'northeast');
exportgraphics(f, fullfile(out, 'sit_stand_gyro_sway.png'), 'Resolution', 150);
close(f);

%% 座位と立位の、骨盤の回転の支点までの距離（第9段）
addpath(fullfile(root, 'matlab'), fileparts(mfilename('fullpath')));
data = har.loadData(root);
[B, bn] = bioPool(data.trainRaw);
pv = B(:, bn == "pivot_height_mean") * 100;   % cm
lab = string(data.trainLabels);
f = figure('Visible', 'off', 'Color', hex('#fcfcfb'), 'Position', [0 0 900 480]);
ax = axes(f); hold(ax, 'on');
edges = -20:1:35;
histogram(ax, pv(lab == "Standing"), edges, 'DisplayStyle', 'stairs', 'EdgeColor', blue, 'LineWidth', 2);
histogram(ax, pv(lab == "Sitting"), edges, 'DisplayStyle', 'stairs', 'EdgeColor', orange, 'LineWidth', 2);
ms = median(pv(lab == "Standing")); mi = median(pv(lab == "Sitting"));
xline(ax, ms, '-', sprintf('立位の中央値 %.1f cm', ms), 'Color', ink, 'FontName', font, 'FontSize', 9, ...
    'LabelHorizontalAlignment', 'left');
xline(ax, mi, '-', sprintf('座位の中央値 %.1f cm', mi), 'Color', ink, 'FontName', font, 'FontSize', 9);
styleAxes(ax, font, ink, muted, grid);
xlabel(ax, '骨盤の回転の支点から腰の端末までの距離（cm、窓ごとの推定）'); ylabel(ax, '窓の数');
title(ax, '座位では座面の坐骨が支点になり、端末は支点の約9 cm上で揺れる', 'FontWeight', 'normal');
legend(ax, {'立位', '座位'}, 'Box', 'off', 'TextColor', ink, 'Location', 'northeast');
exportgraphics(f, fullfile(out, 'sit_stand_pivot.png'), 'Resolution', 150);
close(f);
fprintf('FIGURES_DONE\n');
end


function styleAxes(ax, font, ink, muted, gridColor)
set(ax, 'FontName', font, 'FontSize', 10, 'XColor', muted, 'YColor', muted, ...
    'GridColor', gridColor, 'GridAlpha', 1, 'Box', 'off', 'TickDir', 'out', 'Color', 'none');
ax.YGrid = 'on';
ax.Title.Color = ink; ax.Title.FontSize = 12;
ax.XLabel.Color = muted; ax.YLabel.Color = muted;
end


function c = hex(s)
c = sscanf(s(2:end), '%2x')' / 255;
end
