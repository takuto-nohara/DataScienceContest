function predictFcn = fitModel(method, X, Y, opts)
% 分類学習器で選べる手法を、同じ呼び方で学習する。
% 標準化は学習側の行だけで平均・標準偏差を求める（分類学習器の「標準化」と同じ）。
%
%   method : 'tree' 'linear_svm' 'gauss_svm' 'knn' 'subspace_knn' 'bagged_tree'
%            'nnet' 'lda'
%   opts   : 構造体。KernelScale・BoxConstraint を上書きできる
if nargin < 4, opts = struct(); end
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma < 1e-12) = 1;
Z = (X - mu) ./ sigma;
p = size(X, 2);

switch method
    case 'tree'
        mdl = fitctree(Z, Y, 'MaxNumSplits', 100);
    case 'linear_svm'
        t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1);
        mdl = fitcecoc(Z, Y, 'Learners', t, 'Coding', 'onevsone');
    case 'gauss_svm'
        scale = getOpt(opts, 'KernelScale', sqrt(p));
        box = getOpt(opts, 'BoxConstraint', 10);
        t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', scale, ...
            'BoxConstraint', box);
        mdl = fitcecoc(Z, Y, 'Learners', t, 'Coding', 'onevsone');
    case 'knn'
        mdl = fitcknn(Z, Y, 'NumNeighbors', 10, 'DistanceWeight', 'squaredinverse');
    case 'subspace_knn'
        mdl = fitcensemble(Z, Y, 'Method', 'Subspace', 'Learners', 'knn', ...
            'NumLearningCycles', 30, 'NPredToSample', max(1, ceil(p / 2)));
    case 'bagged_tree'
        mdl = fitcensemble(Z, Y, 'Method', 'Bag', 'NumLearningCycles', 100, ...
            'Learners', templateTree('MaxNumSplits', size(Z, 1) - 1));
    case 'nnet'
        mdl = fitcnet(Z, Y, 'LayerSizes', 100, 'Lambda', 1e-4, 'IterationLimit', 1000);
    case 'lda'
        mdl = fitcdiscr(Z, Y, 'DiscrimType', 'linear', 'Gamma', 0.01);
    otherwise
        error('Unknown method %s', method);
end
predictFcn = @(Xnew) predict(mdl, (Xnew - mu) ./ sigma);
end


function v = getOpt(opts, name, default)
if isfield(opts, name), v = opts.(name); else, v = default; end
end
