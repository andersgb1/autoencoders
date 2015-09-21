function layer = create_layer(insize, hidsize, transfcn, W, b, trainfcn, varargin)


%% Parse inputs
p = inputParser;
p.CaseSensitive = false;
% Set opts
p.addOptional('Name', 'Layer')
p.parse(varargin{:});
% Get opts
name = p.Results.Name;

layer = network;

% Define topology
layer.numInputs = 1;
layer.numLayers = 1;
layer.inputConnect(1,1) = 1;
layer.outputConnect = 1;
layer.biasConnect = 1;

% Set values for labels
layer.name = name;
layer.layers{1}.name = name;

% Copy parameters from inputs
layer.input.size = insize;
layer.layers{1}.size = hidsize;
layer.layers{1}.transferFcn = transfcn;
layer.IW{1,1} = W;
    
% Biases
layer.b{1} = b;

% Set a training function
layer.trainFcn = trainfcn;

% Set the input
layerStruct = struct(layer);
% TODO
%     networkStruct = struct(net);
%     layerStruct.inputs{1} = networkStruct.inputs{1};
layer = network(layerStruct);

end