function output = prop(W, b, transfer, input)

numLayers = length(W);

% Layer outputs and derivatives
o = cell(numLayers, 1);
% do = cell(numLayers, 1);
% Forward propagate
for i = 1:numLayers
    if i == 1
        a = bsxfun(@plus, W{i} * input, b{i});
    else
        a = bsxfun(@plus, W{i} * o{i-1}, b{i});
    end
    o{i} = feval(transfer{i}, a);
%     do{i} = feval(transfer{i}, 'dn', a);
end

output = o{end};