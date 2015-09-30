function [c,mu] = train_pca(X, varargin)
% TRAIN_PCA  Perform principal component analysis
%   [c,mu] = TRAIN_PCA(X) finds the principal components of the data in X,
%   expected to be row-major, which means individual observations are
%   placed in rows. The output value c contains the PCA coefficients and mu
%   contains the mean.
%
%   [c,mu] = TRAIN_PCA(X, num_components) returns a PCA only for the
%   num_components with highest variance.
%
%   See also PROJECT_PCA, REPROJECT_PCA.
%

if nargin == 1
    [c,~,~,~,~,mu] = pca(X);
else
    [c,~,~,~,~,mu] = pca(X, 'NumComponents', varargin{1});
end
end