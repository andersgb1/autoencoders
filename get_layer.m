function layer = get_layer(net, idx, varargin)
% Get the encoder part of an autoencoder as a Network instance
%
% This is just a copy of the private function Autoencoder.getEncoder
%

if nargin == 2
    name = 'Layer';
else
    name = varargin{1};
end

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

% Copy parameters from input network
if idx == 1
    % Special case: getting the input layer
    layer.inputs{1}.size = net.inputs{1}.size;
    layer.layers{1}.size = net.layers{1}.size;
    layer.layers{1}.transferFcn = net.layers{1}.transferFcn;
    layer.IW{1,1} = net.IW{1};
else
    layer.inputs{1}.size = net.layers{idx-1}.size;
    layer.layers{1}.size = net.layers{idx}.size;
    layer.layers{1}.transferFcn = net.layers{idx}.transferFcn;
    layer.IW{1,1} = net.LW{idx,idx-1};
end
    
% Biases
layer.b{1} = net.b{idx};

% Set a training function
layer.trainFcn = net.trainFcn;

% Set the input
layerStruct = struct(layer);
% TODO
if idx == 1
    networkStruct = struct(net);
    layerStruct.inputs{1} = networkStruct.inputs{1};
end
layer = network(layerStruct);

end