% Verify the contest environment using synthetic data only.
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'artifacts');
if ~isfolder(out), mkdir(out); end
diary(fullfile(out, 'environment-check.log'));
cleanupDiary = onCleanup(@() diary('off'));
fprintf('MATLAB: %s\n', version);
assert(strcmp(version('-release'), '2026a'), 'Contest requires R2026a.');
assert(contains(version, 'Update 5'), 'Contest recommends Update 5.');
ver;
features = {'MATLAB', 'Neural_Network_Toolbox', ...
    'Distrib_Computing_Toolbox', 'Statistics_Toolbox', 'Signal_Toolbox'};
for k = 1:numel(features)
    [ok, msg] = license('checkout', features{k});
    assert(ok, 'License checkout failed for %s: %s', features{k}, msg);
    fprintf('License checkout OK: %s\n', features{k});
end

% Exercise actual functionality rather than just checking installation folders.
rng(224224);
x = randn(48, 6);
y = categorical(repelem((1:6)', 8));
tree = fitctree(x, y);
assert(numel(predict(tree, x)) == size(x, 1));
a = dlarray(single(x));
assert(isequal(size(extractdata(relu(a))), size(x)));
assert(numel(findpeaks(sin((0:127) * 0.3))) > 0);
pool = gcp('nocreate');
createdPool = isempty(pool);
if createdPool, pool = parpool('Processes', 2); end
future = parfeval(pool, @sum, 1, 1:128);
assert(fetchOutputs(future) == sum(1:128));
if createdPool, delete(pool); end

% GPU is optional: record availability, but CPU development must work.
try
    count = gpuDeviceCount('available');
    fprintf('Available GPUs: %d\n', count);
    if count > 0
        device = gpuDevice;
        fprintf('GPU: %s\n', device.Name);
        b = gpuArray(single(1:128));
        assert(gather(sum(b)) == single(sum(1:128)));
        fprintf('GPU computation OK\n');
    end
catch exception
    fprintf('Optional GPU check: %s\n', exception.message);
end
fprintf('ENVIRONMENT_CHECK_PASSED\n');
