function stage2b_sitstand(root)
% 座位と立位だけを取り出し、2クラスでどこまで分けられるかを調べる（上限の見積もり）。
% 6クラスの誤りのほとんどがこの組なので、ここの上限が全体の上限になる。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
load(fullfile(out, 'pool.mat'), 'Xtr', 'Ytr', 'names', 'family', 'gtr');

rows = ismember(Ytr, categorical(["Sitting", "Standing"]));
X = Xtr(rows, :);
Y = removecats(Ytr(rows));
g = gtr(rows);
fprintf('sitting %d / standing %d / recordings %d\n', sum(Y == 'Sitting'), ...
    sum(Y == 'Standing'), numel(unique(g)));

% 姿勢の角度（重力方向から求めた前後・左右の傾き、度）
gx = X(:, names == "gravity_dir_x");
gy = X(:, names == "gravity_dir_y");
gz = X(:, names == "gravity_dir_z");
angles = [atan2d(gy, gx), atan2d(gz, gx)];

sets = struct();
sets.posture3 = find(family == "gravity_dir");
sets.posture_angles = [];
sets.posture_plus_mag = [find(family == "gravity_dir"), find(names == "acc_mag_mean")];
sets.sway = find(ismember(family, ["gyro_std", "gyro_mag", "gyro_absmean", "band_frac", "acc_std"]));
sets.posture_plus_sway = [find(family == "gravity_dir"), find(names == "acc_mag_mean"), sets.sway];
sets.pool_all = 1:numel(names);

methods = ["gauss_svm", "knn", "subspace_knn", "bagged_tree", "nnet", "tree"];
result = {};
for s = string(fieldnames(sets))'
    cols = sets.(s);
    Xs = X(:, cols);
    if s == "posture_angles", Xs = angles; end
    for m = methods
        acc = groupedCV(Xs, Y, g, char(m), 3);
        fprintf('%-18s %-13s p=%2d  sit/stand grouped CV %.4f (errors %.0f)\n', s, m, ...
            size(Xs, 2), acc, (1 - acc) * numel(Y));
        result(end + 1, :) = {s, m, size(Xs, 2), acc, (1 - acc) * numel(Y)}; %#ok<AGROW>
    end
end
T = cell2table(result, 'VariableNames', {'featureSet', 'method', 'nFeatures', ...
    'groupedCvAccuracy', 'expectedErrors'});
writetable(T, fullfile(out, 'stage2b_sitstand_ceiling.csv'));

% 姿勢の角度の分布（図とレポート用）
A = table(Y, angles(:, 1), angles(:, 2), X(:, names == "acc_mag_mean"), ...
    X(:, names == "gyro_std_x"), 'VariableNames', ...
    {'label', 'pitch_deg', 'roll_deg', 'acc_mag_mean', 'gyro_std_x'});
writetable(A, fullfile(out, 'stage2b_sitstand_posture.csv'));
fprintf('STAGE2B_DONE\n');
end
