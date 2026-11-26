function make_slide_figures(root)
% 本番発表スライド用の図を書き出す（presentation/images/final_*.png）。
%   final_confusion_compare.png : 検証用407試行での混同行列を4つのモデルで並べる
%   final_confusion_model.png   : 本番モデルの混同行列（記録単位の交差検証の平均と、検証用407試行）
%   final_confusion_cv.png / final_confusion_verification.png : 上の2つを見出しなしで1枚ずつ
% 行が正解、列が予測（図には軸の名前を書かず、スライドの本文で説明する）。対角（正解）は青、対角の外（誤り）は橙で、色の濃さは行ごとの割合。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(mfilename('fullpath')));
end
ev = fullfile(root, 'evaluation', 'feature_study');
out = fullfile(root, 'presentation', 'images');
names = ["歩行", "上り", "下り", "座位", "立位", "臥位"];

panels = {'stage15_confusion_organizer12_gauss_svm.csv', '主催者の12列 ＋ ガウスSVM';
    'stage15_confusion_ergonomic32_knn.csv', '32列 ＋ k近傍法';
    'stage15_confusion_ergonomic32_bagged_tree.csv', '32列 ＋ バギング木';
    'stage15_confusion_ergonomic32_gauss_svm.csv', '32列 ＋ ガウスSVM'};
f = figure('Visible', 'off', 'Color', [0.984 0.984 0.976], 'Units', 'pixels', 'Position', [0 0 1500 1500]);
t = tiledlayout(f, 2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
for k = 1:4
    C = table2array(readtable(fullfile(ev, panels{k, 1}), 'ReadRowNames', true));
    drawMatrix(nexttile(t), C, names, sprintf('%s　%d/407', panels{k, 2}, trace(C)));
end
exportgraphics(f, fullfile(out, 'final_confusion_compare.png'), 'Resolution', 150);
close(f);

cv = readmatrix(fullfile(ev, 'final_cv_confusion_mean.csv'));
if size(cv, 2) > 6, cv = cv(:, end - 5:end); end
vf = table2array(readtable(fullfile(ev, 'stage15_confusion_ergonomic32_gauss_svm.csv'), 'ReadRowNames', true));
f = figure('Visible', 'off', 'Color', [0.984 0.984 0.976], 'Units', 'pixels', 'Position', [0 0 1800 800]);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
drawMatrix(nexttile(t), cv, names, '学習用の交差検証');
drawMatrix(nexttile(t), vf, names, sprintf('本番の検証用　%d/407', trace(vf)));
exportgraphics(f, fullfile(out, 'final_confusion_model.png'), 'Resolution', 150);
close(f);

% 分類割合のスライド用に1枚ずつ（見出しはスライドの本文に書くので、図には付けない）
single = {cv, 'final_confusion_cv.png'; vf, 'final_confusion_verification.png'};
for k = 1:2
    f = figure('Visible', 'off', 'Color', [0.984 0.984 0.976], 'Units', 'pixels', 'Position', [0 0 800 800]);
    t = tiledlayout(f, 1, 1, 'Padding', 'compact');
    drawMatrix(nexttile(t), single{k, 1}, names, '');
    exportgraphics(f, fullfile(out, single{k, 2}), 'Resolution', 150);
    close(f);
end
fprintf('FIGURES_DONE\n');
end


function drawMatrix(ax, C, names, ttl)
n = numel(names);
share = C ./ max(sum(C, 2), 1);
blue = [0.910 0.941 0.984; 0.122 0.306 0.604];
orange = [0.984 0.929 0.890; 0.761 0.329 0.106];
empty = [0.957 0.961 0.969];
hold(ax, 'on');
for i = 1:n
    for j = 1:n
        if i == j
            c = blue(1, :) + (blue(2, :) - blue(1, :)) * share(i, j);
        elseif C(i, j) > 0
            c = orange(1, :) + (orange(2, :) - orange(1, :)) * min(1, 0.35 + 3 * share(i, j));
        else
            c = empty;
        end
        rectangle(ax, 'Position', [j - 0.5 + 0.02, i - 0.5 + 0.02, 0.96, 0.96], 'FaceColor', c, ...
            'EdgeColor', 'none', 'Curvature', 0.06);
        lum = 0.2126 * c(1) + 0.7152 * c(2) + 0.0722 * c(3);
        txtColor = [0.122 0.161 0.216];
        if lum < 0.45, txtColor = [0.984 0.984 0.976]; end
        v = C(i, j);
        if abs(v - round(v)) < 1e-9, s = sprintf('%d', round(v)); else, s = sprintf('%.1f', v); end
        if v == 0, s = '0'; txtColor = [0.62 0.64 0.68]; end
        text(ax, j, i, s, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontName', 'BIZ UDPGothic', 'FontSize', 17, 'FontWeight', 'bold', 'Color', txtColor);
    end
end
set(ax, 'YDir', 'reverse', 'XLim', [0.5 n + 0.5], 'YLim', [0.5 n + 0.5], 'XTick', 1:n, 'YTick', 1:n, ...
    'XTickLabel', names, 'YTickLabel', names, 'TickLength', [0 0], 'FontName', 'BIZ UDPGothic', ...
    'FontSize', 16, 'XAxisLocation', 'top', 'Color', 'none', 'XColor', [0.29 0.33 0.39], 'YColor', [0.29 0.33 0.39]);
axis(ax, 'square');
box(ax, 'off');
title(ax, ttl, 'FontName', 'BIZ UDPGothic', 'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.122 0.161 0.216]);
end
