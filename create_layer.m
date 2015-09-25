function layer = create_layer(insize, hidsize, transfcn, W, b, trainfcn, varargin)


%% Parse inputs
p = inputParser;
p.CaseSensitive = false;
% Set opts
p.addParameter('Name', 'Layer')
p.parse(varargin{:});
% Get opts
name = p.Results.Name;

% Initialize net
layer = network;
layer.name = name;

% Dimensions
layer.numInputs = 1;
layer.numLayers = 1;

% Connections
layer.inputConnect(1,1) = 1;
layer.outputConnect = 1;
layer.biasConnect = 1;

% Subobjects
layer.input.size = insize;
layer.layers{1}.name = name;
layer.layers{1}.size = hidsize;
layer.layers{1}.transferFcn = transfcn;

% Weight and bias values
layer.IW{1,1} = W;
layer.b{1} = b;

% Functions
layer.divideFcn = 'dividetrain';
layer.plotFcns = {'plotperform'};
layer.plotParams = {nnetParam}; % Dummy?
layer.trainFcn = trainfcn;

% Set the input
layerStruct = struct(layer);
% TODO
%     networkStruct = struct(net);
%     layerStruct.inputs{1} = networkStruct.inputs{1};
layer = network(layerStruct);

end