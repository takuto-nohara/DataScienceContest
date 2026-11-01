root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
rng(17); raw=randn(9,128,6); sizes=[12 32 20]; sets={'stats12','temporal32','accel20'};
for k=1:3
    [x,names]=har.features(raw,sets{k});
    assert(isequal(size(x),[9 sizes(k)]) && numel(unique(names))==sizes(k));
    % The same trial must be identical alone, in another batch, or permuted.
    assert(max(abs(x(4,:)-har.features(raw(4,:,:),sets{k})),[],'all')<1e-10);
    assert(max(abs(x([8 2],:)-har.features(raw([8 2],:,:),sets{k})),[],'all')<1e-10);
    zero=har.features(zeros(1,128,6),sets{k}); assert(all(isfinite(zero)));
end
f=har.features(raw,'accel20');raw(:,:,4:6)=1000*randn(9,128,3);
assert(isequal(f,har.features(raw,'accel20')), 'Accelerometer-only features used gyroscope data.');
r=har.metrics(["a";"a";"b";"b"],["a";"b";"b";"b"],["a";"b"]);
assert(isequal(r.confusion,[1 1;0 2]) && r.accuracy==0.75);
assert(abs(r.macroF1-(2/3+0.8)/2)<1e-12 && r.balancedAccuracy==0.75);
rejected=false;try,har.metrics("a","unknown",["a";"b"]);catch,rejected=true;end;assert(rejected);
rejected=false;try,har.features(nan(1,128,6),'stats12');catch,rejected=true;end;assert(rejected);
rejected=false;try,har.features(zeros(1,127,6),'stats12');catch,rejected=true;end;assert(rejected);
data=har.loadData(root);protocol=har.protocol(root,data);
assert(isempty(intersect(protocol.fitRows,protocol.validationRows)));
rejected=false;try,har.evaluate(root,{'proto_missing'});catch exception,rejected=contains(exception.message,'Unknown requested model ID');end
assert(rejected,'Evaluator silently accepted an unknown model ID.');
folders=dir(fullfile(root,'model','proto_*','trained.mat'));
for k=1:numel(folders)
    saved=load(fullfile(folders(k).folder,folders(k).name),'bundle');b=saved.bundle;
    assert(all(ismember(b.fitRows,protocol.fitRows)));
    batch=har.predict(b,data.testRaw(1:8,:,:));
    for j=1:8,assert(batch(j)==har.predict(b,data.testRaw(j,:,:)));end
    [x,~]=har.features(data.trainRaw(b.fitRows,:,:),b.spec.featureSet);
    assert(max(abs(mean(x,1)-b.center))<1e-10, 'Scaler not fitted on training rows.');
end
fprintf('CHECKS_PASSED: features, row independence, sensor isolation, metrics, invalid-input rejection, protocol, saved-model inference (%d models)\n',numel(folders));
