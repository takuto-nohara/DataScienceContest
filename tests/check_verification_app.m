% 本番配布の提出前チェックアプリ（DScontest_submitModelVerifier）と同じ手順で、
% 本番候補のモデルを検証用データ（407窓）に通す。
%   前処理関数(total_acc_x, ..., body_gyro_z) → trainedModel.predictFcn → testActivity と比べる
% アプリは正解率0.5以上で「提出可能」とする。ここではその条件と、出力の形も確かめる。
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'submission', 'ergonomic32_svm'));
v = load(fullfile(root, 'data', 'production', 'Verification', 'rawSensorData_verification.mat'));
m = load(fullfile(root, 'model', 'final_ergonomic32_svm', 'trained.mat'));
X = Pretreatment_MATLABcontest_Advanced(v.total_acc_x, v.total_acc_y, v.total_acc_z, ...
    v.body_gyro_x, v.body_gyro_y, v.body_gyro_z);
assert(isa(X, 'double') && isreal(X) && isequal(size(X), [numel(v.testActivity), 32]) && all(isfinite(X(:))), ...
    'Pretreatment output must be a finite real double N x 32 matrix.');
yfit = m.trainedModel.predictFcn(X);
accuracy = sum(v.testActivity == yfit) / numel(v.testActivity);
fprintf('verification app check: %d/%d correct (%.4f)\n', sum(v.testActivity == yfit), numel(yfit), accuracy);
assert(accuracy >= 0.5, 'The check app would reject this model.');
disp(confusionmat(v.testActivity, yfit));
fprintf('CHECK_VERIFICATION_DONE\n');
