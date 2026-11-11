function ActivityDataTrain = Pretreatment_MATLABcontest_Advanced( ...
    total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z)
% 【下書き】主催者の公式テンプレートが届く前に、要項の記述から再現した枠である。
% 公式ファイルを受け取ったら、下の2つの目印に挟まれた部分だけを、公式ファイルの
% 同じ目印の間へ貼り付ける。関数名・入力・出力変数名は公式ファイルのものを使う。
%
% 入力: 6つのセンサ変数。それぞれ (試行数)×128 の行列
% 出力: ActivityDataTrain  (試行数)×32 の実数行列。1行が1試行

%ここから下を編集してよい

% 規則への対応:
%   ・各行は、その試行の128点だけから計算する (ほかの試行の値は使わない)
%   ・行の順序と数は入力のまま
%   ・使う関数は MATLAB 本体と Statistics and Machine Learning Toolbox のみ
%   ・目印の外を編集しないで済むよう、ローカル関数は使わず無名関数で書く
% 計算の内容は har.features の 'spectral32' と同じである。

sampleRateHz = 50;
signals = cat(3, total_acc_x, total_acc_y, total_acc_z, ...
    body_gyro_x, body_gyro_y, body_gyro_z);
trialCount = size(signals, 1);
sampleCount = size(signals, 2);

% 列 1-12: 6信号それぞれの平均と標準偏差。
% 加速度の平均は端末の傾き、標準偏差は揺れの大きさを表す。
axisMean = reshape(mean(signals, 2), trialCount, 6);
axisStd = reshape(std(signals, 0, 2), trialCount, 6);

% 列 13-20: 加速度と角速度の大きさの平均・標準偏差・最大・変化の激しさ。
% 大きさは端末の向きに左右されにくい。
changeRateRms = @(signal) sqrt(mean((diff(signal, 1, 2) * sampleRateHz) .^ 2, 2));
summarize = @(signal) [mean(signal, 2), std(signal, 0, 2), ...
    max(signal, [], 2), changeRateRms(signal)];
accelerationMagnitude = sqrt(sum(signals(:, :, 1:3) .^ 2, 3));
gyroscopeMagnitude = sqrt(sum(signals(:, :, 4:6) .^ 2, 3));
magnitudeStats = [summarize(accelerationMagnitude), summarize(gyroscopeMagnitude)];

% 列 21-26: 同じセンサの中での軸どうしの相関 (x-y, y-z, z-x)。
% どちらかの軸が変化しない試行では、相関を 0 とする。
axisPairs = [1 2; 2 3; 3 1; 4 5; 5 6; 6 4];
correlations = zeros(trialCount, size(axisPairs, 1));
for pair = 1:size(axisPairs, 1)
    a = signals(:, :, axisPairs(pair, 1)) - axisMean(:, axisPairs(pair, 1));
    b = signals(:, :, axisPairs(pair, 2)) - axisMean(:, axisPairs(pair, 2));
    denominator = sqrt(sum(a .^ 2, 2) .* sum(b .^ 2, 2));
    r = sum(a .* b, 2) ./ max(denominator, eps);
    r(denominator < 1e-12) = 0;
    correlations(:, pair) = r;
end

% 列 27-32: 0 Hz を除く全パワーのうち、0.5〜3 Hz が占める割合。
% 歩行の周期はおおむねこの帯域に入る。
positiveBins = 1:floor(sampleCount / 2);
frequencyHz = positiveBins * sampleRateHz / sampleCount;
power = abs(fft(signals - mean(signals, 2), [], 2)) .^ 2;
positivePower = power(:, positiveBins + 1, :);
inBand = frequencyHz >= 0.5 & frequencyHz <= 3;
bandPower = sum(positivePower(:, inBand, :), 2);
totalPower = max(sum(positivePower, 2), eps);
bandFraction = reshape(bandPower ./ totalPower, trialCount, 6);

ActivityDataTrain = [axisMean, axisStd, magnitudeStats, correlations, bandFraction];

%ここまでを編集してよい
end
