function [augmented, labels] = augment(raw, labels, gains)
% Preserve each trial's gravity direction; vary motion amplitude only.
assert(isvector(gains) && all(isfinite(gains)) && all(gains>0),'Augmentation gains must be finite and positive.');
originalLabels=string(labels(:));augmented=zeros(size(raw,1)*numel(gains),128,6);labels=strings(size(augmented,1),1);
n=size(raw,1);gravity=mean(raw(:,:,1:3),2);body=raw(:,:,1:3)-gravity;
for k=1:numel(gains)
    rows=(k-1)*n+(1:n);augmented(rows,:,1:3)=gravity+gains(k)*body;
    augmented(rows,:,4:6)=gains(k)*raw(:,:,4:6);labels(rows)=originalLabels;
end
end
