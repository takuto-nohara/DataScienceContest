function inspect_verification(root)
% 本番配布の検証用データ（rawSensorData_verification.mat）の中身を確かめる。
% 変数名・大きさ・行動ごとの窓数と、学習用・テスト用の窓との重なり（同じ窓、64点の共有）を数える。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
v = load(fullfile(root, 'data', 'production', 'Verification', 'rawSensorData_verification.mat'));
disp(v);
f = fieldnames(v);
for k = 1:numel(f)
    x = v.(f{k});
    fprintf('%-20s class %-12s size %s\n', f{k}, class(x), mat2str(size(x)));
end
data = har.loadData(root);
vr = har.readVerification(root);
fprintf('verification windows %d\n', size(vr.raw, 1));
disp(countcats(categorical(vr.labels, data.classes))');
allRaw = cat(1, data.trainRaw, data.testRaw, vr.raw);
nTr = size(data.trainRaw, 1); nTe = size(data.testRaw, 1); nV = size(vr.raw, 1);
flat = reshape(allRaw, size(allRaw, 1), []);
d = pdist2(flat(nTr + nTe + 1:end, :), flat(1:nTr + nTe, :), 'euclidean', 'Smallest', 1);
fprintf('verification windows identical to a train/test window: %d\n', sum(d < 1e-9));
g = recordingGroups(allRaw);
src = [ones(nTr, 1); 2 * ones(nTe, 1); 3 * ones(nV, 1)];
gv = unique(g(src == 3));
shared = arrayfun(@(r) any(g == r & src < 3), gv);
fprintf('verification recordings %d, linked to train/test recordings %d\n', numel(gv), sum(shared));
fprintf('verification windows in linked recordings %d\n', sum(ismember(g(src == 3), gv(shared))));
fprintf('INSPECT_DONE\n');
end
