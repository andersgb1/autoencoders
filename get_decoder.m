function decoder = get_decoder(autoenc)
% Get the decoder part of an autoencoder as a Network instance
%
% This is just a copy of the private function Autoencoder.getDecoder
%

decoder = network;

% Define topology
decoder.numInputs = 1;
decoder.numLayers = 1;
decoder.inputConnect(1,1) = 1;
decoder.outputConnect = 1;
decoder.biasConnect = 1;

% Set values for labels
decoder.name = 'Decoder';
decoder.layers{1}.name = 'Decoder';

% Copy parameters from input network
decoder.inputs{1}.size = autoenc.HiddenSize;
decoder.layers{1}.size = autoenc.network.inputs{1}.size;
decoder.layers{1}.transferFcn = autoenc.DecoderTransferFunction;
decoder.IW{1,1} = autoenc.DecoderWeights;
decoder.b{1} = autoenc.DecoderBiases;

% Set a training function
decoder.trainFcn = autoenc.network.trainFcn;

% Set the output
decoderStruct = struct(decoder);
networkStruct = struct(autoenc.network);
decoderStruct.outputs{end} = networkStruct.outputs{end};
decoder = network(decoderStruct);
end