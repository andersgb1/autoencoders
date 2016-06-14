function [dW,db,f] = backprop(W, b, transfer, input, target, loss, varargin)
% BACKPROP  Compute gradient of a feed-forward neural network.
%   [dW,db] = BACKPROP(W, b, transfer, input, target, ...) computes the
%   gradient of a feed-forward network given by corresponding cell lists of
%   weights (W), biases (b) and transfer functions (transfer)
%
%   [dW,db,f] = BACKPROP(...) also outputs the loss
%
%   Name value pair options (default value):
%
%       'DropoutRate' (0): Set this to a number > 0 and < 1 for dropout
%       for the hidden layers, where 0 means no dropout.
%
%       'Normalization' ('full'): normalization factor. If 'full', then the
%       loss/derivatives are divided by the total number of elements in
%       target/output - this is the MATLAB default. If 'batch', then we
%       divide by the number of columns

% Get opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('DropoutRate', 0, @isfloat)
p.addParameter('Normalization', 'full', @ischar)
p.parse(varargin{:});
dropout = p.Results.DropoutRate;
assert(dropout < 1, 'Dropout rate must be < 1!');
normalization = p.Results.Normalization; 

numLayers = length(W);
% numWeightElements = sum(cellfun(@numel, W)) + sum(cellfun(@numel, b));

%% Forward pass
% Layer outputs and derivatives
o = cell(numLayers, 1);
do = cell(numLayers, 1);
% Forward propagate
for i = 1:numLayers
    if i == 1
        a = bsxfun(@plus, W{i} * input, b{i});
    else
        a = bsxfun(@plus, W{i} * o{i-1}, b{i});
    end
    o{i} = feval(transfer{i}, a);
    do{i} = feval(transfer{i}, 'dn', a);

%     % TODO: This is the derivative in the softmax case - and it's slow as
%     % hell on GPU
%     if strcmp(transfer{i}, 'softmax')
%         [dim,N] = size(o{i});
%         tmp = cell(1,N);
%         for j=1:N
%             oj = o{i}(:,j);
%             tmp{j} = gpuArray(zeros(dim,dim));
%             for r=1:dim
%                 for c=1:dim
%                     if r==c
%                         tmp{j}(r,c) = oj(r)*(1-oj(r));
%                     else
%                         tmp{j}(r,c) = -oj(r)*oj(c);
%                     end
%                 end
%             end
%         end
%     end
    
    % Dropout hidden layers
    % https://gist.github.com/ottokart/ebd3d32438c13a62ea3c
    if dropout > 0 && i < numLayers
        mask = binornd(1, 1-dropout, size(a));
        o{i} = mask .* o{i} / (1-dropout);
        do{i} = mask .* do{i} / (1-dropout);
    end
end

%% Backward pass
% Output error
if any(strcmp(loss, {'crossentropy', 'log', 'binary_crossentropy', 'crossentropy_binary'}))
    assert(any(strcmp(transfer{end}, {'logsig', 'softmax'})), 'Cross-entropy loss function requires logistic or softmax output units!')
end
if nargout < 3
    [~,delta] = backprop_loss(target, o{end}, loss, 'Normalization', normalization);
else
    [f,delta] = backprop_loss(target, o{end}, loss, 'Normalization', normalization);
end

% Backpropagate
dW = cell(1, numLayers);
db = cell(1, numLayers);
for i = numLayers:-1:1
    % Delta
    if i == numLayers % Output layer
        if strcmp(transfer{i}, 'softmax') % Softmax outputs cells with a Jacobian per sample
            for j = 1:length(do{i}), delta(:,j) = do{i}{j} * delta(:,j); end
        else
            delta = do{i} .* delta;
        end
    else % Input or hidden layer
        if strcmp(transfer{i}, 'softmax'), error('Softmax transfer function only supported for the output layer!'); end
        delta = do{i} .* (W{i+1}' * delta);
    end
    
    % Weight update
    if i > 1 % Hidden or output layer
        dW{i} = delta * o{i-1}';
    else % Input layer
        dW{i} = delta * input';
    end
    
    % Bias update
    db{i} = sum(delta, 2);
end
