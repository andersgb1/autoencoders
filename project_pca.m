function Y = project_pca(X, c, mu)
% PROJECT_PCA  Project data to an existing PCA model
%   Y = PROJECT_PCA(X, c, mu) projects the data in X to the PCA model given
%   by the coefficient matrix c and the mean mu.
%
%   See also TRAIN_PCA, REPROJECT_PCA.
%

Y = (X - repmat(mu, size(X,1), 1)) * c;
end