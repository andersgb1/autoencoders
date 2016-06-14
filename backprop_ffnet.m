function grad = backprop_ffnet(ffnet, input, target, varargin)
% BACKPROP_FFNET  Compute gradient of a feed-forward neural network.
%   grad = BACKPROP_FFNET(ffnet, input, target, ...) computes the gradient of the
%   network in ffnet.
%
%   Name value pair options (default value):
%
%       'Loss' (empty): loss function. If empty, use the loss provided by
%       the network. Otherwise it can be one of these:
%       'mse', 'mae', 'crossentropy', 'log', 'crossentropy_binary',
%       'binary_crossentropy'.
%
%       'DropoutRate' (0): Set this to a number > 0 and < 1 for dropout
%       for the hidden layers, where 0 means no dropout.
%
%       'Normalization' ('full'): normalization factor. If 'full', then the
%       loss/derivatives are divided by the total number of elements in
%       target/output - this is the MATLAB default. If 'batch', then we
%       divide by the number of columns.

% Get opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Loss', '', @ischar)
p.addParameter('DropoutRate', 0, @isfloat)
p.addParameter('Normalization', 'full', @ischar)
p.parse(varargin{:});

loss = p.Results.Loss;
if isempty(loss)
    loss = ffnet.performFcn;
end
dropout = p.Results.DropoutRate;
normalization = p.Results.Normalization;

%% Get weights/biases and run backprop
[W,b,transfer] = get_ffnet_info(ffnet);
[dW,db] = backprop(W, b, transfer, input, target, loss,...
    'DropoutRate', dropout,...
    'Normalization', normalization);

grad = zeros(sum(cellfun(@numel, dW)) + sum(cellfun(@numel, db)), 1);
count = 1;
for k=1:length(dW)
    bnum = numel(db{k});
    grad(count:count+bnum-1) = db{k};
    count = count + bnum;
    Wnum = numel(dW{k});
    grad(count:count+Wnum-1) = reshape(dW{k}, Wnum, 1);
    count = count + Wnum;
end

return

% %% Forward pass
% % Layer outputs and derivatives
% o = cell(ffnet.numLayers, 1);
% do = cell(ffnet.numLayers, 1);
% % Forward propagate
% for i = 1:ffnet.numLayers
%     if i == 1
%         a = bsxfun(@plus, ffnet.IW{1} * input, ffnet.b{1});
%     else
%         a = bsxfun(@plus, ffnet.LW{i,i-1} * o{i-1}, ffnet.b{i});
%     end
%     o{i} = feval(ffnet.layers{i}.transferFcn, a);
%     do{i} = feval(ffnet.layers{i}.transferFcn, 'dn', a);
%     % Dropout hidden layers
%     % https://gist.github.com/ottokart/ebd3d32438c13a62ea3c
%     if dropout > 0 && i < ffnet.numLayers
%         mask = binornd(1, 1-dropout, size(a));
%         o{i} = mask .* o{i} / (1-dropout);
%         do{i} = mask .* do{i} / (1-dropout);
%     end
% end
% 
% %% Backward pass
% % % Output error
% if any(strcmp(loss, {'crossentropy', 'log', 'binary_crossentropy', 'crossentropy_binary'}))
%     assert(any(strcmp(ffnet.layers{end}.transferFcn, {'logsig', 'softmax'})), 'Cross-entropy loss function requires logistic or softmax output units!')
% end
% [~,delta] = backprop_loss(target, o{end}, loss, 'Normalization', normalization);
% 
% % Backpropagate
% grad = zeros(ffnet.numWeightElements, 1);
% idx = 1;
% for i = ffnet.numLayers:-1:1
%     % Delta
%     if i == ffnet.numLayers % Output layer
%         if strcmp(ffnet.layers{i}.transferFcn, 'softmax') % Softmax outputs cells
%             for j = 1:length(do{i}), delta(:,j) = do{i}{j} * delta(:,j); end
%         else
%             delta = do{i} .* delta;
%         end
%     else % Input or hidden layer
%         if strcmp(ffnet.layers{i}.transferFcn, 'softmax'), error('Softmax transfer function only supported for the output layer!'); end
%         delta = do{i} .* (ffnet.LW{i+1, i}' * delta);
%     end
%     
%     % Weight update
%     if i > 1 % Hidden or output layer
%         dw = delta * o{i-1}';
%     else % Input layer
%         dw = delta * input';
%     end
%     
%     % Bias update
%     db = sum(delta, 2);
%     
%     % Collect updates
%     numwb = numel(dw) + numel(db);
%     grad((idx+numwb-1):-1:idx) = [db ; dw(:)];
%     idx = idx + numwb;
% end
% grad = flipud(grad);
