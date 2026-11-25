% 本番提出ファイルを作り、提出前チェックアプリと同じ確認をする。
% 公式ライブスクリプト（Official_MATLABcontest_Advanced_workplace.mlx）の手順を、そのまま順に行う。
%   1. 学習用データを読み、前処理関数で ActivityDataTrain（2941×32）を作り、形を確かめる
%   2. 本番候補のモデル（model/final_ergonomic32_svm/trained.mat の trainedModel）を使う
%   3. 提出作業２：save("trainedModel_"+teamname+".mat","trainedModel")
%   4. 提出作業３：copyfile で前処理関数を Pretreatment_MATLABcontest_Advanced_<teamname>.m にする
%   5. 提出作業４：チェックアプリ（DScontest_submitModelVerifier）と同じ確認
%      ファイル名の形、変数名 trainedModel、検証用407窓の正解率0.5以上
% このフォルダで実行する: cd submission/seitailab; build_submission
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
cd(here);
teamname = "seitailab"; % ファイル用チーム名

%% 1. 前処理とデータ形式のチェック（公式と同じ）
load(fullfile(root, 'data', 'production', 'Rawdata Files', 'rawSensorData_train.mat'));
ActivityDataTrain = Pretreatment_MATLABcontest_Advanced(total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z);
numtrials = height(trainActivity);
validateattributes(ActivityDataTrain, {'double'}, {'real', 'size', [numtrials, 32]});
disp('ActivityDataTrain は正しい次元数のデータです。');

%% 2. モデル
m = load(fullfile(root, 'model', 'final_ergonomic32_svm', 'trained.mat'));
trainedModel = m.trainedModel;
% 学習時の前処理と、この前処理関数の出力が一致することを確かめる（学習用2941窓）
accTrain = mean(trainedModel.predictFcn(ActivityDataTrain) == trainActivity);
fprintf('training-set accuracy through this Pretreatment: %.4f\n', accTrain);

%% 3-4. 提出作業２・３（公式と同じ）
save("trainedModel_"+teamname+".mat","trainedModel")
copyfile("Pretreatment_MATLABcontest_Advanced.m",...
    "Pretreatment_MATLABcontest_Advanced_"+teamname+".m");

%% 5. 提出作業４：チェックアプリと同じ確認
preFile = "Pretreatment_MATLABcontest_Advanced_" + teamname + ".m";
modelFile = "trainedModel_" + teamname + ".mat";
assert(~isempty(regexp(preFile, "^Pretreatment_MATLABcontest_Advanced_\w+\.m$", 'once')), 'Pretreatment file name');
assert(~isempty(regexp(modelFile, "^trainedModel_\w+\.mat$", 'once')), 'model file name');
assert(ismember("trainedModel", who('-file', fullfile(here, modelFile))), 'variable name trainedModel');
TargetFunc = str2func(extractBefore(preFile, "."));
Submitted = load(fullfile(here, modelFile));
V = load(fullfile(root, 'data', 'production', 'Verification', 'rawSensorData_verification.mat'));
X = TargetFunc(V.total_acc_x, V.total_acc_y, V.total_acc_z, V.body_gyro_x, V.body_gyro_y, V.body_gyro_z);
validateattributes(X, {'double'}, {'real', 'finite', 'size', [height(V.testActivity), 32]});
yfit = Submitted.trainedModel.predictFcn(X);
accuracy = sum(V.testActivity == yfit) / numel(V.testActivity);
fprintf('check app: %d/%d (%.4f) -> %s\n', sum(V.testActivity == yfit), numel(yfit), accuracy, ...
    string(accuracy >= 0.5));
assert(accuracy >= 0.5, 'The check app would not pass this model.');

% 1試行だけ渡しても同じ値になること（審査では1試行ずつ届く場合もある）
one = zeros(5, 32);
for i = 1:5
    one(i, :) = TargetFunc(V.total_acc_x(i, :), V.total_acc_y(i, :), V.total_acc_z(i, :), ...
        V.body_gyro_x(i, :), V.body_gyro_y(i, :), V.body_gyro_z(i, :));
end
fprintf('single-trial max abs difference: %.3g\n', max(abs(one - X(1:5, :)), [], 'all'));
assert(max(abs(one - X(1:5, :)), [], 'all') < 1e-10, 'single-trial output differs');
v = ver('MATLAB');
fprintf('MATLAB %s %s\n', v.Version, v.Release);
fprintf('BUILD_SUBMISSION_DONE\n');
