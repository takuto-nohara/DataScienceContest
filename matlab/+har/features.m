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
if strcmp(featureSet,'shape32')
    direction=means(:,1:3)./max(sqrt(sum(means(:,1:3).^2,2)),eps);
    body=raw(:,:,1:3)-mean(raw(:,:,1:3),2);
    vertical=sum(body.*reshape(direction,n,1,3),3);
    horizontal=sqrt(max(sum(body.^2,3)-vertical.^2,0));
    rotation=sqrt(sum(raw(:,:,4:6).^2,3));
    x=[x,reshape(mean(abs(raw(:,:,4:6)),2),n,3)];names=[names,sensor(4:6)+"_mean_absolute"];
    signals={vertical,horizontal,rotation};signalNames=["vertical","horizontal","rotation"];
    for j=1:3
        q=signals{j};qCentered=q-mean(q,2);variance=mean(qCentered.^2,2);
        skew=mean(qCentered.^3,2)./max(variance.^1.5,eps);
        kurt=mean(qCentered.^4,2)./max(variance.^2,eps);
        spectrum=abs(fft(qCentered,[],2)).^2;spectrum=spectrum(:,2:65);hz=(1:64)*50/128;
        low=sum(spectrum(:,hz>=0.5 & hz<=3),2)./max(sum(spectrum,2),eps);
        x=[x,std(q,0,2),sqrt(mean((diff(q,1,2)*50).^2,2)),skew,kurt,prctile(q,90,2)-prctile(q,10,2),low]; %#ok<AGROW>
        names=[names,signalNames(j)+"_"+["std","jerk_rms","skewness","kurtosis","p90_minus_p10","low_band_power_fraction"]]; %#ok<AGROW>
    end
    derivative=diff(vertical,1,2)*50;
    x=[x,max(derivative,[],2),min(derivative,[],2)];names=[names,"vertical_jerk_max","vertical_jerk_min"];
    % stats12 (12) + gyro absolute means (3) + 3*6 (18) + 2 = 35.
    % Remove gyro means (noise-sensitive) to keep the contest's 32-feature limit.
    keep=[1:3,7:35];x=x(:,keep);names=names(keep);
    assert(size(x,2)==32 && all(isfinite(x(:))));return;
end
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
if ismember(string(featureSet),["stable26","gravity32","spectral32","distribution32","window32"])
    columns=[1:20,27:32];x=x(:,columns);names=names(columns);
    switch featureSet
        case 'stable26'
            % Drop all dominant-bin frequencies: low-motion windows have unstable peaks.
        case 'gravity32'
            direction=means(:,1:3)./max(sqrt(sum(means(:,1:3).^2,2)),eps);
            jerk=reshape(sqrt(mean((diff(raw(:,:,1:3),1,2)*50).^2,2)),n,3);
            x=[x,direction,jerk];names=[names,"gravity_direction_"+["x","y","z"],sensor(1:3)+"_jerk_rms"];
        case 'spectral32'
            frequencies=(1:64)*50/128;
            positive=power(:,2:65,:);
            ratio=reshape(sum(positive(:,frequencies>=0.5 & frequencies<=3,:),2)./max(sum(positive,2),eps),n,6);
            x=[x,ratio];names=[names,sensor+"_power_fraction_0p5_to_3_hz"];
        case 'distribution32'
            spread=reshape(prctile(raw,75,2)-prctile(raw,25,2),n,6);
            x=[x,spread];names=[names,sensor+"_iqr"];
        case 'window32'
            drift=reshape(mean(raw(:,65:128,:),2)-mean(raw(:,1:64,:),2),n,6);
            x=[x,drift];names=[names,sensor+"_second_half_minus_first_half_mean"];
    end
elseif strcmp(featureSet, 'accel20')
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
