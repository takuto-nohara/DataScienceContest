function data = loadData(root)
% All prototypes consume this one common data directory.
base = fullfile(root, 'data', 'prestudy', 'Rawdata Files');
trainFile = fullfile(base, 'rawSensorData_train.mat');
testFile = fullfile(base, 'rawSensorData_test.mat');
assert(isfile(trainFile) && isfile(testFile), 'Download organizer data into data/prestudy first.');
train = load(trainFile);
test = load(testFile);
names = {'total_acc_x','total_acc_y','total_acc_z','body_gyro_x','body_gyro_y','body_gyro_z'};
data.classes = ["Walking"; "WalkingUpstairs"; "WalkingDownstairs"; "Sitting"; "Standing"; "Laying"];
data.trainLabels = string(train.trainActivity(:));
data.testLabels = string(test.testActivity(:));
assert(all(ismember(data.trainLabels, data.classes)) && all(ismember(data.testLabels, data.classes)), 'Unknown activity label.');
data.trainRaw = zeros(numel(data.trainLabels), 128, 6);
data.testRaw = zeros(numel(data.testLabels), 128, 6);
for k = 1:6
    a = train.(names{k}); b = test.([names{k} '_test']);
    assert(isequal(size(a), [numel(data.trainLabels),128]) && isequal(size(b), [numel(data.testLabels),128]), 'Invalid sensor shape.');
    assert(isreal(a) && isreal(b) && all(isfinite(a(:))) && all(isfinite(b(:))), 'Invalid sensor values.');
    data.trainRaw(:,:,k) = a; data.testRaw(:,:,k) = b;
end
data.fingerprint = struct('train', har.hashFile(trainFile), 'test', har.hashFile(testFile));
end
