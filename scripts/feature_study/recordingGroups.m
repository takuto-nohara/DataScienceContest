function recording = recordingGroups(raw)
% 64点を共有する窓どうしをつなぎ、連続した記録ごとに番号を付ける。
% 同じ記録の窓が学習側と評価側に分かれると、重なりの分だけ正解率が水増しされる。
% 交差検証はこの番号の単位で分ける（記録単位の交差検証）。
n = size(raw, 1);
head = reshape(raw(:, 1:64, :), n, []);
tail = reshape(raw(:, 65:128, :), n, []);
[follower, leader] = find(pdist2(head, tail) < 1e-9);
recording = conncomp(graph(leader, follower, [], n))';
end
