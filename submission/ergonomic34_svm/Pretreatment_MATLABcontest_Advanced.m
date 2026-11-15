function ActivityDataTrain = Pretreatment_MATLABcontest_Advanced( ...
    total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z)
% 【本番候補】人体工学の役割ごとに選んだ34列を計算する前処理関数。
% 公式ファイルを受け取ったら、下の2つの目印に挟まれた部分だけを、公式ファイルの
% 同じ目印の間へ貼り付ける。関数名・入力・出力変数名は公式ファイルのものを使う。
%
% 入力: 6つのセンサ変数。それぞれ (試行数)×128 の行列（50 Hz）
% 出力: ActivityDataTrain  (試行数)×34 の実数行列。1行が1試行
% 列の選び方と根拠は reports/feature-study-ja.md にある。

%ここから下を編集してよい

% 規則への対応:
%   ・各行は、その試行の128点だけから計算する（ほかの試行の値は使わない）
%   ・行の順序と数は入力のまま
%   ・使う関数は MATLAB 本体のみ（skewness も使わず、式で計算する）
%   ・目印の外を編集しないで済むよう、ローカル関数は使わず無名関数で書く
% 計算の内容は scripts/feature_study/featurePool.m の同名の列と同じである（列 14-16 は
% その常用対数）。最後に、学習データで求めた固定の定数で標準化する。

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

% --- 骨盤のねじれのなめらかさ（列 31-32）------------------------------------
% 鉛直軸（x軸）まわりの角速度の1階差分と2階差分のばらつき。歩行や階段で骨盤を
% 回す動きが、なめらかか、細かく切り替わるかを表す。差分はサンプル間の差のまま。
dGyroX = diff(gyro(:, :, 1), 1, 2);
ddGyroX = diff(gyro(:, :, 1), 2, 2);
twistChange = [std(dGyroX, 0, 2), std(ddGyroX, 0, 2)];

% --- 座面と支点：骨盤の回転の支点までの距離と、前後の揺れの周波数（列 33-34）------
% 重力方向を鉛直とし、水平の2軸（e1 は y 軸を水平面に写したもの、e2 = 鉛直 × e1）に
% 加速度と角速度を回す。腰の装着角度に左右されない形で揺れを測るため。
% 列 33：水平の加速度 = 支点までの距離 × 角加速度 ＋ 重力 × 傾き角 ＋ ゆっくりした変化
% を窓ごとに最小二乗で当てはめ、角加速度の係数（m）を支点までの距離とする（2軸の平均）。
% 立位では骨盤は股関節（ベルトとほぼ同じ高さ）を軸に回るので距離はほぼ 0（中央値 0.02 m）。
% 座位では骨盤が座面の坐骨を支点に転がるので、ベルトは支点の約 0.1 m 上にある（0.09 m）。
% 列 34：前後方向の水平の揺れの、0.2〜5 Hz での平均周波数（体幹加速度による動揺の指標）。
G = 9.80665;
ev = gravityDir;
e1 = repmat([0 1 0], n, 1);
e1 = e1 - sum(e1 .* ev, 2) .* ev;
e1 = e1 ./ max(sqrt(sum(e1 .^ 2, 2)), eps);
e2 = cross(ev, e1, 2);
project = @(s, e) sum(s .* reshape(e, n, 1, 3), 3);
h1 = project(body, e1) * G;                 % 前後の水平の揺れ [m/s^2]
h2 = project(body, e2) * G;
smooth5 = @(s) movmean(s, 5, 2);
tt = ((1:L) - (L + 1) / 2) / fs;
pairs = {smooth5(h2), smooth5(project(gyro, e1)), -1; smooth5(h1), smooth5(project(gyro, e2)), 1};
pivotHeight = zeros(n, 2);
for q = 1:2
    w = pairs{q, 2};
    theta = cumsum(w - mean(w, 2), 2) / fs;
    alpha = gradient(w) * fs;
    for i = 1:n
        A = [ones(L, 1), tt', theta(i, :)', alpha(i, :)'];
        b = pinv(A) * pairs{q, 1}(i, :)';
        pivotHeight(i, q) = pairs{q, 3} * b(4);
    end
end
fullHz = (0:L / 2) * fs / L;
swayBand = fullHz >= 0.2 & fullHz <= 5;
pH1 = abs(fft(h1 - mean(h1, 2), [], 2)) .^ 2;
pH1 = pH1(:, swayBand);
swayFreqAP = (pH1 * fullHz(swayBand)') ./ max(sum(pH1, 2), eps);
support = [mean(pivotHeight, 2), swayFreqAP];

% --- 標準化（列 1-34）-------------------------------------------------------
% 学習データ（2941窓＋拡張8通り）で求めた固定の平均と標準偏差で割る。どの試行も
% 同じ定数を使うので、ほかの試行の値には依存しない。分類学習器では「データを標準化」
% を切る。全クラスで1回だけ標準化すると、2クラスの組ごとに標準化するより、静止時の
% 小さな揺れの差が保たれる。定数は scripts/feature_study/write_scaling.m が書き込む。
featureMean = [0.78533380932230035, 0.025485375925599973, 0.080690021802598863, 1.0327396323371218, 0.12903419374238248, 0.084210657227779234, 0.071962600227180568, 0.13637335314245985, 0.134509515153967, 0.090185786574194279, 0.051986520927793219, 0.073753723107349567, 1.1279172557180486, -1.1362507279172407, -1.1024478451306188, -1.2608895893185628, 0.39108320115711243, 2.7674054956510155, 2.1448635549355917, 1.755632183077467, 8.0770192422070117, -9.1018940371181731, 0.44427607899091376, 0.40635685211531619, 0.46101606163942405, 3.0705140919566287, -0.1285479738018955, -0.10485679728502495, -0.0048243686761887004, 0.24084414056769987, 0.093966989316975655, 0.086971282389877225, 0.055826960224698623, 2.0612259279428726];
featureStd = [0.35375281951016746, 0.3660804426090859, 0.34197760951311251, 0.020838819359483687, 0.14765647740086221, 0.08732340819339475, 0.075923000662055581, 0.15761704756321324, 0.15570834832525618, 0.089829642064039927, 0.052355191756790501, 0.081476308339620468, 0.60872029699410424, 0.82177510272620691, 0.78695877711365403, 0.78974606687760018, 0.40166257322941085, 3.1112233629614829, 2.3048679371839302, 1.9704978894277325, 9.6301732714210786, 10.426030877137977, 0.26220951932009823, 0.19161831946067814, 0.20193270268133356, 2.6984757247629991, 0.30717711247868135, 0.41966858256629697, 0.38368900391328875, 0.57648469236275679, 0.10817649030001497, 0.1031171485865893, 0.071251366939208208, 0.63415181306590185];
features = [gravityDir, accMagMean, intensity, impact, rhythm, coordination, twistChange, support];
features(~isfinite(features)) = 0;                     % 値が変化しない試行への備え
features = (features - featureMean) ./ featureStd;

ActivityDataTrain = features;

%ここまでを編集してよい
end
