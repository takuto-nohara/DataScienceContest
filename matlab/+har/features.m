function [x, names] = features(raw, featureSet)
% Stateless, row-independent features; no batch statistics or learned parameters.
assert(isnumeric(raw) && isreal(raw) && size(raw,2)==128 && size(raw,3)==6, 'Expected N x 128 x 6 real sensor array.');
assert(all(isfinite(raw(:))), 'Nonfinite input.');
n = size(raw,1);
sensor = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"];
means = reshape(mean(raw,2), n,6);
deviations = reshape(std(raw,0,2), n,6);
x = [means, deviations];
names = [sensor+"_mean", sensor+"_std"];
if strcmp(featureSet, 'stats12'), return; end
magAcc = sqrt(sum(raw(:,:,1:3).^2,3));
magGyro = sqrt(sum(raw(:,:,4:6).^2,3));
for signal = {magAcc, magGyro}
    s = signal{1};
    x = [x, mean(s,2),std(s,0,2),max(s,[],2),sqrt(mean((diff(s,1,2)*50).^2,2))]; %#ok<AGROW>
end
names = [names, "acc_magnitude_"+["mean","std","max","jerk_rms"], "gyro_magnitude_"+["mean","std","max","change_rms"]];
% Remove DC before estimating the strongest nonzero frequency.
power = abs(fft(raw-mean(raw,2), [],2)).^2;
[~, peak] = max(power(:,2:65,:), [],2);
frequency = reshape(peak,n,6)*50/128;
flat = deviations < 1e-12;
frequency(flat) = 0;
x = [x, frequency]; names = [names, sensor+"_dominant_hz"];
pairs = [1 2;2 3;3 1;4 5;5 6;6 4];
for k=1:6
    a = raw(:,:,pairs(k,1))-means(:,pairs(k,1));
    b = raw(:,:,pairs(k,2))-means(:,pairs(k,2));
    denominator = sqrt(sum(a.^2,2).*sum(b.^2,2));
    r = sum(a.*b,2)./max(denominator,eps);
    r(denominator<1e-12) = 0;
    x(:,end+1) = r; names(end+1) = sensor(pairs(k,1))+"_"+sensor(pairs(k,2))+"_correlation";
end
if strcmp(featureSet, 'accel20')
    columns = [1:3,7:9,13:16,21:23,27:29];
    x = [x(:,columns),reshape(max(raw(:,:,1:3),[],2),n,3)];
    names = [names(columns),sensor(1:3)+"_max"];
    spectrum = abs(fft(magAcc-mean(magAcc,2),[],2)).^2;
    spectrum = spectrum(:,2:65);
    probability = spectrum./max(sum(spectrum,2),eps);
    x(:,end+1) = -sum(probability.*log(max(probability,eps)),2)/log(64);
    names(end+1) = "acc_magnitude_spectral_entropy";
else
    assert(strcmp(featureSet,'temporal32'), 'Unknown feature set: %s',featureSet);
end
assert(all(isfinite(x(:))) && size(x,2)==numel(names), 'Invalid extracted features.');
end
