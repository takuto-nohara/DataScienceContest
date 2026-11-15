function ActivityDataTrain = Pretreatment_MATLABcontest_Advanced( ...
    total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z)
% 【本番候補】人体工学の役割ごとに選んだ30列（＋ゼロ埋め2列）を計算する前処理関数。
% 公式ファイルを受け取ったら、下の2つの目印に挟まれた部分だけを、公式ファイルの
% 同じ目印の間へ貼り付ける。関数名・入力・出力変数名は公式ファイルのものを使う。
%
% 入力: 6つのセンサ変数。それぞれ (試行数)×128 の行列（50 Hz）
% 出力: ActivityDataTrain  (試行数)×32 の実数行列。1行が1試行
% 列の選び方と根拠は reports/feature-study-ja.md にある。

%ここから下を編集してよい

% 規則への対応:
%   ・各行は、その試行の128点だけから計算する（ほかの試行の値は使わない）
%   ・行の順序と数は入力のまま
%   ・使う関数は MATLAB 本体のみ（skewness も使わず、式で計算する）
%   ・目印の外を編集しないで済むよう、ローカル関数は使わず無名関数で書く
% 計算の内容は scripts/feature_study/featurePool.m の同名の列と同じである。

fs = 50;                                    % サンプリング周波数 [Hz]
acc = cat(3, total_acc_x, total_acc_y, total_acc_z);   % 重力を含む加速度 [g]
gyro = cat(3, body_gyro_x, body_gyro_y, body_gyro_z);  % 角速度 [rad/s]
n = size(acc, 1);
L = size(acc, 2);
flat = @(v) reshape(v, n, []);              % n×1×3 を n×3 にする
rmsOf = @(v) sqrt(mean(v .^ 2, 2));

% --- 姿勢：重力の向き（列 1-4）---------------------------------------------
% 窓の平均加速度は重力にほぼ等しい。その向きは体幹（腰）の傾きを表す。
% 座位では骨盤が後ろに傾くため、立位より端末が約15°傾く（座位/立位の主な手がかり）。
accMean = flat(mean(acc, 2));
gNorm = max(sqrt(sum(accMean .^ 2, 2)), eps);
gravityDir = accMean ./ gNorm;
accMagnitude = sqrt(sum(acc .^ 2, 3));
accMagMean = mean(accMagnitude, 2);

% --- 動きの強さ（列 5-17）---------------------------------------------------
% 静止と動作を分け、階段下りの大きな上下動を捉える。
% 回転の揺れ（列 14-16）は対数で持つ。立って静止していると、鉛直軸（x軸）まわりの
% 骨盤のねじれが座位の約3倍になる（中央値 0.022 と 0.007 rad/s）。左右の脚への体重の
% 移しかえで骨盤が回るためと考えられる。座位では坐骨と太ももが座面に接し、ねじれが
% 抑えられる。前後・左右の傾きの揺れ（y・z軸）は座位と立位で差がない。差は「比」で
% 現れ、歩行の揺れ（約0.5 rad/s）と同じ目盛りでは埋もれるので、対数にして比を差に変える。
% 上下＝重力方向への射影、水平＝それと直交する成分。端末の向きに左右されにくい。
body = acc - mean(acc, 2);
vertical = sum(body .* reshape(gravityDir, n, 1, 3), 3);
horizontal = sqrt(max(sum(body .^ 2, 3) - vertical .^ 2, 0));
xyMotion = sqrt(body(:, :, 1) .^ 2 + body(:, :, 2) .^ 2);
gyroMagnitude = sqrt(sum(gyro .^ 2, 3));

intensity = [flat(std(acc, 0, 2)), std(accMagnitude, 0, 2), std(vertical, 0, 2), ...
    mean(horizontal, 2), std(horizontal, 0, 2), std(xyMotion, 0, 2), ...
    max(acc(:, :, 1), [], 2), ...                       % x軸＝鉛直方向の最大加速度
    log10(max(flat(std(gyro, 0, 2)), 1e-9)), ...         % 回転の揺れ（対数、上の説明）
    mean(gyroMagnitude, 2)];

% --- 衝撃：加速度の時間変化（列 18-22）------------------------------------
% 着地の衝撃は躍度（加速度の変化率）に現れる。階段下りは踵からの着地で大きい。
jerk = diff(acc, 1, 2) * fs;
verticalJerk = diff(vertical, 1, 2) * fs;
impact = [flat(rmsOf(jerk)), max(verticalJerk, [], 2), min(verticalJerk, [], 2)];

% --- リズム：歩調（列 23-26）----------------------------------------------
% 歩行の周期は 0.5〜3 Hz に入る。上り階段は歩調が遅い。
bins = 1:floor(L / 2);
hz = bins * fs / L;
inBand = hz >= 0.5 & hz <= 3;
bandFraction = @(signal) sum(signal(:, inBand), 2) ./ max(sum(signal, 2), eps);
spectrumOf = @(signal) abs(fft(signal - mean(signal, 2), [], 2)) .^ 2;
pAccX = spectrumOf(acc(:, :, 1));
pAccY = spectrumOf(acc(:, :, 2));
pGyroX = spectrumOf(gyro(:, :, 1));
pMag = spectrumOf(accMagnitude);
pMag = pMag(:, bins + 1);
[~, peak] = max(pMag, [], 2);
stepHz = hz(peak)';
stepHz(std(accMagnitude, 0, 2) < 1e-9) = 0;
rhythm = [bandFraction(pAccX(:, bins + 1)), bandFraction(pAccY(:, bins + 1)), ...
    bandFraction(pGyroX(:, bins + 1)), stepHz];

% --- 協調：軸どうしの相関と上下動の非対称性（列 27-30）---------------------
% 歩行では骨盤の回旋と傾きが連動する。階段では体を持ち上げる・下ろす動きで偏りが出る。
correlate = @(a, b) sum((a - mean(a, 2)) .* (b - mean(b, 2)), 2) ./ max(sqrt( ...
    sum((a - mean(a, 2)) .^ 2, 2) .* sum((b - mean(b, 2)) .^ 2, 2)), eps);
vCentered = vertical - mean(vertical, 2);
vVar = mean(vCentered .^ 2, 2);
verticalSkew = mean(vCentered .^ 3, 2) ./ max(vVar .^ 1.5, eps);
verticalSkew(vVar < 1e-24) = 0;
coordination = [correlate(acc(:, :, 1), acc(:, :, 2)), ...
    correlate(gyro(:, :, 2), gyro(:, :, 3)), correlate(gyro(:, :, 3), gyro(:, :, 1)), ...
    verticalSkew];

ActivityDataTrain = [gravityDir, accMagMean, intensity, impact, rhythm, coordination, ...
    zeros(n, 2)];                                       % 未使用の列 31-32 はゼロ
ActivityDataTrain(~isfinite(ActivityDataTrain)) = 0;   % 値が変化しない試行への備え

%ここまでを編集してよい
end
