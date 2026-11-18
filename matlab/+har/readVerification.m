function v = readVerification(root)
% 本番配布（DScontest_Advanced_Playground.zip）の提出前チェックアプリに同梱された
% 検証用データ rawSensorData_verification.mat を読む。407窓、ラベル付き。
% ファイルは data/production/Verification に置く（アプリの .mltbx から取り出したもの）。
%   v.raw    N×128×6 の波形（加速度 x/y/z、角速度 x/y/z）
%   v.labels N×1 の行動名
%   v.sha256 ファイルの SHA-256
file = fullfile(root, 'data', 'production', 'Verification', 'rawSensorData_verification.mat');
assert(isfile(file), 'Extract rawSensorData_verification.mat into data/production/Verification first.');
s = load(file);
names = {'total_acc_x', 'total_acc_y', 'total_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z'};
n = size(s.total_acc_x, 1);
v.raw = zeros(n, 128, 6);
for k = 1:6
    x = s.(names{k});
    assert(isequal(size(x), [n, 128]) && isreal(x) && all(isfinite(x(:))), 'Invalid sensor values.');
    v.raw(:, :, k) = x;
end
v.labels = string(s.testActivity(:));
v.sha256 = har.hashFile(file);
end
