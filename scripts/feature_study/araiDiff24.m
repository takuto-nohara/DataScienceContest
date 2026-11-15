function [F, names] = araiDiff24(raw)
% セミナー資料で99.9%と報告されたコードの24列を、そのまま再現する。
%   1階差分の平均絶対値・標準偏差・最大絶対値、2階差分の標準偏差を、
%   total_acc x/y/z と body_gyro x/y/z の6チャンネルについて求める。
% 差分は時間で割っていない（サンプル間の差のまま）。姿勢（平均）の列は含まない。
ch = ["X", "Y", "Z", "GX", "GY", "GZ"];
F = zeros(size(raw, 1), 24);
for c = 1:6
    s = raw(:, :, c);
    d = diff(s, 1, 2);
    dd = diff(s, 2, 2);
    F(:, c) = mean(abs(d), 2);
    F(:, 6 + c) = std(d, [], 2);
    F(:, 12 + c) = max(abs(d), [], 2);
    F(:, 18 + c) = std(dd, [], 2);
end
names = ["d" + ch + "_mean", "d" + ch + "_std", "d" + ch + "_max", "dd" + ch + "_std"];
end
