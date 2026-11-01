function protocol = protocol(root, data)
file = fullfile(root,'evaluation','protocol.json');
if isfile(file)
    protocol = jsondecode(fileread(file));
    assert(isequal(protocol.fingerprint,data.fingerprint), 'Dataset changed: create a new protocol version.');
else
    rng(224224,'twister');
    partition = cvpartition(categorical(data.trainLabels),'Holdout',0.25);
    protocol = struct('version','protocol_v1','seed',224224, ...
        'fingerprint',data.fingerprint,'fitRows',find(training(partition)), ...
        'validationRows',find(test(partition)), 'classes',data.classes, ...
        'selectionMetric','validation_accuracy','testRole','Fixed external comparison only; do not tune on these labels', ...
        'limitation','Random stratified windows; no subject IDs or recording IDs provided. Overlap leakage cannot be excluded. Not an unseen-subject estimate.');
    har.writeJson(file,protocol);
end
a=protocol.fitRows(:); b=protocol.validationRows(:);
assert(isempty(intersect(a,b)) && isequal(sort([a;b]),(1:numel(data.trainLabels))'), 'Invalid partition.');
end
