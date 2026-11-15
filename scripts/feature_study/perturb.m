function out = perturb(raw, kind, amount, seed)
% 本番で起こりうる「ずれ」を波形に加える。学習データの拡張と、頑健性の試験の両方に使う。
%
%   kind = 'rotate' : 端末の装着角度のずれ。ランダムな軸のまわりに amount 度回す
%                     （加速度と角速度に同じ回転をかける）
%          'gain'   : 動きの強さ。重力成分を保って動きだけを amount 倍
%          'tempo'  : 歩調。時間を amount 倍に伸縮（1.1 なら10%速い動き）
%          'noise'  : センサ雑音。標準偏差 amount（加速度 g、角速度 rad/s）
if nargin < 4, seed = 1; end
stream = RandStream('twister', 'Seed', seed);
n = size(raw, 1);
out = raw;
switch kind
    case 'rotate'
        for i = 1:n
            axis = randn(stream, 3, 1);
            axis = axis / norm(axis);
            R = axang(axis, deg2rad(amount));
            out(i, :, 1:3) = reshape(squeeze(raw(i, :, 1:3)) * R', 1, 128, 3);
            out(i, :, 4:6) = reshape(squeeze(raw(i, :, 4:6)) * R', 1, 128, 3);
        end
    case 'gain'
        g = mean(raw(:, :, 1:3), 2);
        out(:, :, 1:3) = g + amount * (raw(:, :, 1:3) - g);
        out(:, :, 4:6) = amount * raw(:, :, 4:6);
    case 'tempo'
        % 時間軸を amount 倍の速さで読み直す。窓の外へ出る部分は端で折り返す
        t = (0:127) * amount;
        t = mod(t, 2 * 127);
        t(t > 127) = 2 * 127 - t(t > 127);
        for c = 1:6
            out(:, :, c) = interp1(0:127, raw(:, :, c)', t, 'pchip')';
        end
        out(:, :, 4:6) = out(:, :, 4:6) * amount;
    case 'noise'
        out = raw + amount * randn(stream, size(raw));
    otherwise
        error('unknown perturbation');
end
end


function R = axang(u, theta)
K = [0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K ^ 2;
end
