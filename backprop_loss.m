function [f,df] = backprop_loss(target, output, loss, varargin)
% BACKPROP_LOSS  Compute the loss between a target datum and some output,
% and optionally its derivative w.r.t. the output.
%   f = BACKPROP_LOSS(target, output, loss, ...) computes the loss between
%   target and output by the named loss. All data are expected to given in
%   column-major (one sample per column). The loss can be one of these:
%   'mse', 'mae', 'crossentropy', 'log', 'crossentropy_binary',
%   'binary_crossentropy'.
%
%   [f,df] = BACKPROP_LOSS(target, output, loss, ...) also returns the
%   derivative in df.
%
%   Name value pair options (default value):
%
%       'Normalization' ('full'): normalization factor. If 'full', then the
%       loss/derivatives are divided by the total number of elements in
%       target/output - this is the MATLAB default. If 'batch', then we
%       divide by the number of columns

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Normalization', 'full', @ischar)
p.parse(varargin{:});
normalization = p.Results.Normalization; 

%% Get normalization factor
if strcmpi(normalization, 'full')
    N = numel(target);
elseif strcmpi(normalization, 'batch')
    N = size(target, 2);
elseif strcmpi(normalization, 'none')
    N = 1;
else
    error('Unknown normalization method: %s!', normalization);
end

%% Run
switch loss
    case 'mse'
        f = sse(target - output) / N;
        if nargout > 1, df = 2 * (output - target) / N; end
    case 'mae'
        f = sae(target - output) / N;
        if nargout > 1, df = sign(output - target) / N; end
    case {'crossentropy', 'log'}
        f = -sum(sum( target .* log(output + eps) )) / N;
        if nargout > 1, df = -target ./ (output + eps) / N; end
    case {'crossentropy_binary', 'binary_crossentropy'}
        f = -sum(sum( target .* log(output + eps) + (1 - target) .* log(1 - output + eps) )) / N;
        if nargout > 1, df = -(target - output) ./ (output .* (1 - output) + eps) / N; end
    otherwise
        error('Unknown loss function: %s!\n', loss);
end
