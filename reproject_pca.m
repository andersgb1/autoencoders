function X = reproject_pca(Y, c, mu)
% REPROJECT_PCA  Reproject data from an existing PCA model
%   X = REPROJECT_PCA(Y, c, mu) reprojects the data in Y back to the
%   original domain. The data in Y is computed by PROJECT_PCA.
%
%   See also TRAIN_PCA, PROJECT_PCA.
%

X = Y * c' + repmat(mu, size(Y,1) ,1);
end