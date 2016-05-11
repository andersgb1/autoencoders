function grad = backprop(ffnet, input, target, varargin)

% Get opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Loss', '', @ischar)
p.parse(varargin{:});

loss = p.Results.Loss;
if isempty(loss)
    loss = ffnet.performFcn;
end

%% Forward pass
% Layer outputs and derivatives
o = cell(ffnet.numLayers, 1);
do = cell(ffnet.numLayers, 1);
% Forward propagate
for i = 1:ffnet.numLayers
    if i == 1
        a = bsxfun(@plus, ffnet.IW{1} * input, ffnet.b{1});
    else
        a = bsxfun(@plus, ffnet.LW{i,i-1} * o{i-1}, ffnet.b{i});
    end
    o{i} = feval(ffnet.layers{i}.transferFcn, a);
    do{i} = feval(ffnet.layers{i}.transferFcn, 'dn', a);
end

%% Backward pass
% % Output error
if any(strcmp(loss, {'crossentropy', 'log', 'binary_crossentropy', 'crossentropy_binary'}))
    assert(strcmp(ffnet.layers{end}.transferFcn, 'logsig'), 'Cross-entropy loss function requires logistic output units!')
end
[~,delta] = backprop_loss(target, o{end}, loss);

% Backpropagate
grad = zeros(ffnet.numWeightElements, 1);
idx = 1;
for i = ffnet.numLayers:-1:1
    % Delta
    if i == ffnet.numLayers % Output layer
        if strcmp(ffnet.layers{i}.transferFcn, 'softmax') % Softmax outputs cells
            for j = 1:length(do{i}), delta(:,j) = do{i}{j} * delta(:,j); end
        else
            delta = do{i} .* delta;
        end
    else % Input or hidden layer
        if strcmp(ffnet.layers{i}.transferFcn, 'softmax'), error('Softmax transfer function only supported for the output layer!'); end
        delta = do{i} .* (ffnet.LW{i+1, i}' * delta);
    end
    
    % Weight update
    if i > 1 % Hidden or output layer
        dw = delta * o{i-1}';
    else % Input layer
        dw = delta * input';
    end
    
    % Bias update
    db = sum(delta, 2);
    
    % Collect updates
    numwb = numel(dw) + numel(db);
    grad((idx+numwb-1):-1:idx) = [db ; dw(:)];
    idx = idx + numwb;
end
grad = flipud(grad);
