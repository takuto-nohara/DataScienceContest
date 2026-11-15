function stage2c_concepts(root, finalNames, finalMethod)
% 第2c段：人体の動きの「意味のまとまり」ごとに特徴量を抜き、何が必須かを確かめる。
%   姿勢   : 重力の向き（窓の平均＝直流成分を含む列）
%   強さ   : 揺れや回転の大きさ
%   衝撃   : 加速度の時間変化（躍度・2階差分）
%   リズム : 周波数（歩調）
%   協調   : 軸どうしの相関
% 同じ情報を持つ列をすべて抜くので、別の系統が代わりを務めることはない。
% (A) 全71列から抜く、(B) 最終候補のセットから1列ずつ抜く、の2通りで測る。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
out = fullfile(root, 'evaluation', 'feature_study');
load(fullfile(out, 'pool.mat'), 'Xtr', 'Ytr', 'names', 'family', 'gtr');
if isempty(gcp('nocreate')), parpool('Processes', 8); end
pctRunOnAll(sprintf('addpath(''%s'')', strrep(here, '\', '/')));

concept = struct();
concept.posture = ismember(family, ["acc_mean", "gravity_dir", "acc_rms", "acc_maxmin"]) ...
    | names == "acc_mag_mean" | names == "acc_mag_max";
concept.intensity = ismember(family, ["acc_std", "gyro_std", "gyro_mag", "gyro_absmean", ...
    "gyro_maxmin", "acc_maxmin", "xy_motion", "z_change"]) ...
    | ismember(names, ["acc_mag_std", "acc_mag_max", "vertical_std", "horizontal_std", "horizontal_mean"]);
concept.impact = ismember(family, ["acc_jerk", "acc_diff2"]) ...
    | ismember(names, ["acc_mag_jerk_rms", "gyro_mag_change_rms", "vertical_jerk_max", ...
    "vertical_jerk_min", "acc_z_mean_abs_change"]);
concept.rhythm = ismember(family, ["band_frac", "step_freq"]);
concept.coordination = ismember(family, ["acc_corr", "gyro_corr"]) | names == "vertical_skewness";
concept.gyroscope = startsWith(names, "gyro");

pairs = {'static', 'dynamic'; 'Sitting', 'Standing'; 'Laying', 'Sitting'; ...
    'Walking', 'WalkingUpstairs'; 'Walking', 'WalkingDownstairs'; ...
    'WalkingUpstairs', 'WalkingDownstairs'};
methods = ["gauss_svm", "knn", "bagged_tree"];
conceptNames = ["none", string(fieldnames(concept))'];

% (A) 全71列から、まとまりごとに抜く
jobs = combinations(methods, conceptNames);
res = cell(height(jobs), 1);
parfor j = 1:height(jobs)
    keep = true(1, numel(names));
    if jobs.conceptNames(j) ~= "none", keep = ~concept.(jobs.conceptNames(j)); end
    [a, p] = groupedCV(Xtr(:, keep), Ytr, gtr, char(jobs.methods(j)), 3);
    res{j} = [a, sum(keep), pairErrors(Ytr, p, pairs)];
end
A = [jobs, array2table(cell2mat(res), 'VariableNames', ...
    ["groupedCv", "nKept", ("err_" + string(pairs(:, 1)) + "_vs_" + string(pairs(:, 2)))'])];
writetable(A, fullfile(out, 'stage2c_concept_ablation.csv'));
disp(A);

% (B) 最終候補から1列ずつ抜く
if nargin >= 2 && ~isempty(finalNames)
    finalNames = string(finalNames);
    cols = arrayfun(@(s) find(names == s), finalNames);
    drops = [0, 1:numel(cols)];
    res = cell(numel(drops), 1);
    parfor d = 1:numel(drops)
        c = cols;
        if drops(d) > 0, c(drops(d)) = []; end
        [a, p] = groupedCV(Xtr(:, c), Ytr, gtr, finalMethod, 5);
        res{d} = [a, pairErrors(Ytr, p, pairs)];
    end
    dropped = ["(none)", finalNames];
    B = [table(dropped', 'VariableNames', {'droppedFeature'}), array2table(cell2mat(res), ...
        'VariableNames', ["groupedCv", ("err_" + string(pairs(:, 1)) + "_vs_" + string(pairs(:, 2)))'])];
    B.delta = B.groupedCv - B.groupedCv(1);
    writetable(B, fullfile(out, sprintf('stage2c_final_drop_one_%s.csv', finalMethod)));
    disp(B);
end
fprintf('STAGE2C_DONE\n');
end


function e = pairErrors(Y, P, pairs)
static = categorical(["Sitting", "Standing", "Laying"]);
e = zeros(1, size(pairs, 1));
for k = 1:size(pairs, 1)
    if strcmp(pairs{k, 1}, 'static')
        e(k) = sum(ismember(Y, static) ~= ismember(P, static));
    else
        e(k) = sum(Y == pairs{k, 1} & P == pairs{k, 2}) + sum(Y == pairs{k, 2} & P == pairs{k, 1});
    end
end
end
