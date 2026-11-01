root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
d=har.loadData(root);p=har.protocol(root,d);saved=load(fullfile(root,'model/proto_v10/trained.mat'),'bundle');b=saved.bundle;
assert(isequal(b.fitRows(:),p.fitRows(:)) && b.augmentedRows==3*numel(p.fitRows));
prediction=har.predict(b,d.testRaw);assert(all(prediction==d.testLabels),'Development target not achieved.');
rng(92);order=randperm(numel(prediction));permuted=har.predict(b,d.testRaw(order,:,:));
assert(isequal(permuted,prediction(order)),'Predictions depend on row order.');
% Check every window alone, including all five original errors and the new row 155.
for k=1:numel(prediction),assert(har.predict(b,d.testRaw(k,:,:))==prediction(k));end
assert(all(ismember([8;10;105;176;201],find(prediction==d.testLabels))));
% Original baseline artifacts have not been refitted or changed.
for k=1:7
    folder=fullfile(root,'model',sprintf('proto_v%d',k));meta=jsondecode(fileread(fullfile(folder,'manifest.json')));
    assert(strcmp(meta.artifactSHA256,har.hashFile(fullfile(folder,'trained.mat'))));
end
fprintf('REFINEMENT_CHECKS_PASSED: saved proto_v10 201/201, every row alone, shuffled order, fitting-only augmentation, original artifacts unchanged\n');
