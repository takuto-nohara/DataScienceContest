function [x, names] = features(raw, featureSet)
% 6軸のセンサ波形から、分類器に渡す特徴量を計算する。
%
%   raw        : N×128×6 の実数配列 (加速度 x/y/z、角速度 x/y/z の順)
%   featureSet : 特徴量セットの名前 ('stats12'、'spectral32' など)
%   x          : N×(特徴量の数) の行列。1行が1試行に対応する
%   names      : 各列の名前
%
% どの特徴量も、その試行の128点だけから計算する。ほかの試行の値や学習で
% 決めた値は使わないので、1試行だけを渡しても同じ結果になる。
% 標準化は特徴量の計算に含めず、har.train と har.predict が別に行う。

assert(isnumeric(raw) && isreal(raw) && size(raw, 2) == 128 && size(raw, 3) == 6, ...
    'Expected N x 128 x 6 real sensor array.');
assert(all(isfinite(raw(:))), 'Nonfinite input.');
featureSet = char(featureSet);

[means, deviations, axisNames] = axisStatistics(raw);
axisStats = [means, deviations];

% 平均と標準偏差だけのセットと、重力方向を使うセットは、ここで計算を終える。
switch featureSet
    case 'stats12'
        x = axisStats;
        names = axisNames;
        return
    case 'shape32'
        [x, names] = shapeFeatures(raw, means, deviations, axisNames);
        return
end

% 残りのセットが共通で使う部品。
[magnitudeStats, magnitudeNames] = magnitudeStatistics(raw);
power = spectralPower(raw);
[dominantHz, dominantNames] = dominantFrequencies(power, deviations);
[correlations, correlationNames] = axisCorrelations(raw, means);

% 卓越周波数は動きの小さい試行で値が安定しないため、
% temporal32 と accel20 以外では使わない。
stable = [axisStats, magnitudeStats, correlations];
stableNames = [axisNames, magnitudeNames, correlationNames];

switch featureSet
    case 'temporal32'
        x = [axisStats, magnitudeStats, dominantHz, correlations];
        names = [axisNames, magnitudeNames, dominantNames, correlationNames];
    case 'stable26'
        x = stable;
        names = stableNames;
    case 'gravity32'
        [extra, extraNames] = gravityFeatures(raw, means);
        x = [stable, extra];
        names = [stableNames, extraNames];
    case 'spectral32'
        [extra, extraNames] = bandPowerFractions(power);
        x = [stable, extra];
        names = [stableNames, extraNames];
    case 'distribution32'
        [extra, extraNames] = interquartileRanges(raw);
        x = [stable, extra];
        names = [stableNames, extraNames];
    case 'window32'
        [extra, extraNames] = halfWindowDrift(raw);
        x = [stable, extra];
        names = [stableNames, extraNames];
    case 'accel20'
        temporal = [axisStats, magnitudeStats, dominantHz, correlations];
        temporalNames = [axisNames, magnitudeNames, dominantNames, correlationNames];
        [x, names] = accelerationOnlyFeatures(raw, temporal, temporalNames);
    otherwise
        error('Unknown feature set: %s', featureSet);
end

assert(all(isfinite(x(:))) && size(x, 2) == numel(names), ...
    'Invalid extracted features.');
end


function names = sensorNames()
% 6つの信号の名前。列の順序は入力配列の3次元目と同じ。
names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"];
end


function hz = positiveFrequencies()
% FFTの正の周波数 (0 Hz を除く64本) を Hz で返す。50 Hz で128点を取っている。
sampleRateHz = 50;
windowLength = 128;
hz = (1:64) * sampleRateHz / windowLength;
end


function [means, deviations, names] = axisStatistics(raw)
% 6信号それぞれの平均と標準偏差 (12列)。
% 加速度の平均は端末の傾き、標準偏差は揺れの大きさを表す。
trialCount = size(raw, 1);
means = reshape(mean(raw, 2), trialCount, 6);
deviations = reshape(std(raw, 0, 2), trialCount, 6);
names = [sensorNames() + "_mean", sensorNames() + "_std"];
end


function [stats, names] = magnitudeStatistics(raw)
% 加速度と角速度のベクトルの大きさについて、平均・標準偏差・最大・
% 変化の激しさ (8列)。大きさは端末の向きに左右されにくい。
accelerationMagnitude = vectorMagnitude(raw(:, :, 1:3));
gyroscopeMagnitude = vectorMagnitude(raw(:, :, 4:6));

stats = [magnitudeSummary(accelerationMagnitude), ...
    magnitudeSummary(gyroscopeMagnitude)];
names = ["acc_magnitude_" + ["mean", "std", "max", "jerk_rms"], ...
    "gyro_magnitude_" + ["mean", "std", "max", "change_rms"]];
end


function magnitude = vectorMagnitude(threeAxes)
% 3軸の値を、各時刻のベクトルの大きさにまとめる。
magnitude = sqrt(sum(threeAxes .^ 2, 3));
end


function summary = magnitudeSummary(signal)
% 1本の信号の平均・標準偏差・最大・時間変化の二乗平均平方根。
summary = [mean(signal, 2), std(signal, 0, 2), max(signal, [], 2), ...
    changeRateRms(signal)];
end


function rms = changeRateRms(signal)
% 隣り合う点の差を1秒あたりの変化量に直し、その二乗平均平方根を取る。
sampleRateHz = 50;
rms = sqrt(mean((diff(signal, 1, 2) * sampleRateHz) .^ 2, 2));
end


function power = spectralPower(raw)
% 各信号のパワースペクトル。平均 (0 Hz の成分) は先に引いておく。
power = abs(fft(raw - mean(raw, 2), [], 2)) .^ 2;
end


function [hz, names] = dominantFrequencies(power, deviations)
% 6信号それぞれで、パワーが最大になる周波数 (6列)。
trialCount = size(power, 1);
[~, peakBin] = max(power(:, 2:65, :), [], 2);
hz = reshape(peakBin, trialCount, 6) * 50 / 128;

% 値が変化しない信号には卓越周波数がないので 0 とする。
isFlat = deviations < 1e-12;
hz(isFlat) = 0;
names = sensorNames() + "_dominant_hz";
end


function [correlations, names] = axisCorrelations(raw, means)
% 同じセンサの中での軸どうしの相関係数 (加速度3組、角速度3組の6列)。
sensor = sensorNames();
axisPairs = [1 2; 2 3; 3 1; 4 5; 5 6; 6 4];
pairCount = size(axisPairs, 1);

correlations = zeros(size(raw, 1), pairCount);
names = strings(1, pairCount);
for pair = 1:pairCount
    first = axisPairs(pair, 1);
    second = axisPairs(pair, 2);
    a = raw(:, :, first) - means(:, first);
    b = raw(:, :, second) - means(:, second);

    denominator = sqrt(sum(a .^ 2, 2) .* sum(b .^ 2, 2));
    r = sum(a .* b, 2) ./ max(denominator, eps);
    % どちらかの軸が変化しないときは相関を定義できないので 0 とする。
    r(denominator < 1e-12) = 0;

    correlations(:, pair) = r;
    names(pair) = sensor(first) + "_" + sensor(second) + "_correlation";
end
end


function [fractions, names] = bandPowerFractions(power)
% 0 Hz を除く全パワーのうち、0.5〜3 Hz が占める割合 (6列)。
% 歩行の周期はおおむねこの帯域に入る。
trialCount = size(power, 1);
hz = positiveFrequencies();
positivePower = power(:, 2:65, :);
inBand = hz >= 0.5 & hz <= 3;

bandPower = sum(positivePower(:, inBand, :), 2);
totalPower = max(sum(positivePower, 2), eps);
fractions = reshape(bandPower ./ totalPower, trialCount, 6);
names = sensorNames() + "_power_fraction_0p5_to_3_hz";
end


function [x, names] = gravityFeatures(raw, means)
% 重力の向き (加速度の平均を長さ1にしたもの) と、加速度各軸の変化の激しさ。
trialCount = size(raw, 1);
sensor = sensorNames();
direction = gravityDirection(means);
jerk = reshape(changeRateRms(raw(:, :, 1:3)), trialCount, 3);

x = [direction, jerk];
names = ["gravity_direction_" + ["x", "y", "z"], sensor(1:3) + "_jerk_rms"];
end


function direction = gravityDirection(means)
% 加速度の平均を重力の向きとみなし、長さ1のベクトルにする。
gravity = means(:, 1:3);
direction = gravity ./ max(sqrt(sum(gravity .^ 2, 2)), eps);
end


function [spread, names] = interquartileRanges(raw)
% 6信号それぞれの四分位範囲 (6列)。外れ値に強い揺れの指標。
trialCount = size(raw, 1);
spread = reshape(prctile(raw, 75, 2) - prctile(raw, 25, 2), trialCount, 6);
names = sensorNames() + "_iqr";
end


function [drift, names] = halfWindowDrift(raw)
% 後半64点の平均から前半64点の平均を引いた値 (6列)。姿勢の変化を捉える。
trialCount = size(raw, 1);
drift = reshape(mean(raw(:, 65:128, :), 2) - mean(raw(:, 1:64, :), 2), ...
    trialCount, 6);
names = sensorNames() + "_second_half_minus_first_half_mean";
end


function [x, names] = accelerationOnlyFeatures(raw, temporal, temporalNames)
% 加速度だけで作る20列。角速度センサを使えない場合を想定したセット。
trialCount = size(raw, 1);
sensor = sensorNames();

% temporal32 から加速度に関する列だけを取り出す:
% 平均(1-3)、標準偏差(7-9)、大きさの統計(13-16)、卓越周波数(21-23)、相関(27-29)
accelerationColumns = [1:3, 7:9, 13:16, 21:23, 27:29];
axisMaxima = reshape(max(raw(:, :, 1:3), [], 2), trialCount, 3);

% 加速度の大きさのスペクトルが、特定の周波数に集中しているほど小さくなる値。
magnitude = vectorMagnitude(raw(:, :, 1:3));
spectrum = abs(fft(magnitude - mean(magnitude, 2), [], 2)) .^ 2;
spectrum = spectrum(:, 2:65);
probability = spectrum ./ max(sum(spectrum, 2), eps);
entropy = -sum(probability .* log(max(probability, eps)), 2) / log(64);

x = [temporal(:, accelerationColumns), axisMaxima, entropy];
names = [temporalNames(accelerationColumns), sensor(1:3) + "_max", ...
    "acc_magnitude_spectral_entropy"];
end


function [x, names] = shapeFeatures(raw, means, deviations, axisNames)
% 加速度を重力方向 (上下) と水平方向に分け、波形の形を表す32列を作る。
trialCount = size(raw, 1);
sensor = sensorNames();

direction = gravityDirection(means);
body = raw(:, :, 1:3) - mean(raw(:, :, 1:3), 2);
vertical = sum(body .* reshape(direction, trialCount, 1, 3), 3);
horizontal = sqrt(max(sum(body .^ 2, 3) - vertical .^ 2, 0));
rotation = vectorMagnitude(raw(:, :, 4:6));

% 角速度の平均は雑音の影響を受けやすいので、32列に収めるために外す。
% 残すのは加速度の平均(3)と6信号の標準偏差(6)。
x = [means(:, 1:3), deviations];
names = axisNames([1:3, 7:12]);

gyroscopeAbsoluteMean = reshape(mean(abs(raw(:, :, 4:6)), 2), trialCount, 3);
x = [x, gyroscopeAbsoluteMean];
names = [names, sensor(4:6) + "_mean_absolute"];

% 上下・水平・回転の3信号それぞれから6列ずつ。
signals = {vertical, horizontal, rotation};
signalNames = ["vertical", "horizontal", "rotation"];
suffixes = ["std", "jerk_rms", "skewness", "kurtosis", ...
    "p90_minus_p10", "low_band_power_fraction"];
for k = 1:numel(signals)
    x = [x, waveformShape(signals{k})]; %#ok<AGROW>
    names = [names, signalNames(k) + "_" + suffixes]; %#ok<AGROW>
end

% 着地の衝撃は、上下方向の加速度の急な変化として現れる。
sampleRateHz = 50;
verticalJerk = diff(vertical, 1, 2) * sampleRateHz;
x = [x, max(verticalJerk, [], 2), min(verticalJerk, [], 2)];
names = [names, "vertical_jerk_max", "vertical_jerk_min"];

assert(size(x, 2) == 32 && all(isfinite(x(:))));
end


function shape = waveformShape(signal)
% 1本の信号について、揺れの大きさ・変化の激しさ・分布の偏りと尖り・
% 振れ幅・低い周波数の割合を並べる (6列)。
centered = signal - mean(signal, 2);
variance = mean(centered .^ 2, 2);
skewness = mean(centered .^ 3, 2) ./ max(variance .^ 1.5, eps);
kurtosis = mean(centered .^ 4, 2) ./ max(variance .^ 2, eps);

spectrum = abs(fft(centered, [], 2)) .^ 2;
spectrum = spectrum(:, 2:65);
hz = positiveFrequencies();
inBand = hz >= 0.5 & hz <= 3;
lowBandFraction = sum(spectrum(:, inBand), 2) ./ max(sum(spectrum, 2), eps);

shape = [std(signal, 0, 2), changeRateRms(signal), skewness, kurtosis, ...
    prctile(signal, 90, 2) - prctile(signal, 10, 2), lowBandFraction];
end
