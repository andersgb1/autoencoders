function [net,varargout] = train_dbn(X, num_hidden, varargin)
% TRAIN_DBN  Train a deep belief network
%   net = TRAIN_DBN(X, num_hidden, ...) trains a DBN on the data X using a
%   stack of restricted Boltzmann machines. The sizes of the hidden layers
%   is given in the vector num_hidden. The result is returned as a MATLAB
%   network.
%
%   [net,enc] = TRAIN_DBN(...) additionally returns the encoder part of the
%   network.
%
%   [net,enc,dec] = TRAIN_DBN(...) additionally returns the decoder part of
%   the network.
%
%   [net,enc,dec,enc_init] = TRAIN_DBN(...) additionally returns the
%   encoder of the initialized network before fine tuning
%
%   [net,enc,dec,enc_init,dec_init] = TRAIN_DBN(...) additionally returns
%   the decoder of the initialized network before fine tuning
%
%   The data in X is expected to be row-major, meaning that all the feature
%   vectors are stacked as rows on top of each other. If your data is in
%   column-major, use the 'RowMajor' option described below.
%
%   Name value pair options (default value):
%
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units in all layers, can be 'logsig', 'tansig' or 'purelin'
%
%       'OutputFunction' ('logsig'): transfer function for the hidden
%       units in the final layer, can be 'logsig', 'tansig' or 'purelin'.
%       For data compression, it can be an advantage to set this to
%       'purelin', thus implying Gaussian units for the final detectors.
%
%       'VisibleFunction' ('logsig'): transfer function for the visible
%       units in all layers, can be 'logsig', 'tansig' or 'purelin'
%
%       'UnitFunction' ('default'): function determining the type of hidden
%       units, can be 'binary', 'gaussian' or 'default'. When using
%       'default', the type of units depends on the transfer function. For
%       'purelin' the default is 'gaussian', otherwise it is set to
%       'binary'.
%
%       'MaxEpochsInit' (50): number of training iterations for the
%       RBM-based initialization
%
%       'MaxEpochs' (200): number of training iterations for the fine
%       tuning based on backpropagation
%
%       'NumBatches' (100): number of mini-batches considered in each epoch
%
%       'LearningRate' (0.1): learning rate
%
%       'LearningRateFinal' (0.001): learning rate for the final layer
%
%       'Momentum' (0.9): momentum
%
%       'Regularizer' (0.0002): regularizer for the weight update
%
%       'RowMajor' (true): logical specifying whether the observations in X
%       are placed in rows or columns
%
%       'Verbose' (false): logical, set to true to print status messages
%
%       'Visualize' (false): logical, set to true to show status plots
%
%       'Callback' (0): here you can specify a handle to a function, which
%       will be called after every epoch. This is useful if you want to
%       show some kind of progress of where we are. The function handle
%       must take one argument, and then it will be called with the current
%       epoch index.
%
%       See also TRAIN_RBM.

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('HiddenFunction', 'logsig', @ischar)
p.addParameter('OutputFunction', 'logsig', @ischar)
p.addParameter('VisibleFunction', 'logsig', @ischar)
p.addParameter('UnitFunction', 'default', @ischar)
p.addParameter('MaxEpochsInit', 50, @isnumeric)
p.addParameter('MaxEpochs', 200, @isnumeric)
p.addParameter('NumBatches', 100, @isnumeric)
p.addParameter('LearningRate', 0.1, @isfloat)
p.addParameter('LearningRateFinal', 0.001, @isfloat)
p.addParameter('Momentum', 0.9, @isfloat)
p.addParameter('Regularizer', 0.0002, @isfloat)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.parse(varargin{:});
% Get opts
hidden_function = p.Results.HiddenFunction;
output_function = p.Results.OutputFunction;
visible_function = p.Results.VisibleFunction;
unit_function = p.Results.UnitFunction;
max_epochs_init = p.Results.MaxEpochsInit;
max_epochs = p.Results.MaxEpochs;
num_batches = p.Results.NumBatches;
regularizer = p.Results.Regularizer;
learning_rate = p.Results.LearningRate;
learning_rate_final = p.Results.LearningRateFinal;
momentum = p.Results.Momentum;
row_major = p.Results.RowMajor;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
% Transpose data to ensure row-major
if ~row_major, X = X'; end

%% Start pretraining
inputs = X;
enc_init = [];
dec_init = [];
for i = 1:length(num_hidden)
    numhid = num_hidden(i);
    if i == length(num_hidden)
        hidfun = output_function;
        learnrate = learning_rate_final;
    else
        hidfun = hidden_function;
        learnrate = learning_rate;
    end
    
    [enci,deci] = train_rbm(inputs, numhid,...
        'HiddenFunction', hidfun,...
        'VisibleFunction', visible_function,...
        'UnitFunction', unit_function,...
        'MaxEpochs', max_epochs_init,...
        'NumBatches', num_batches,...
        'LearningRate', learnrate,...
        'Momentum', momentum,...
        'Regularizer', regularizer,...
        'Verbose', verbose,...
        'Visualize', (i == 1 && visualize));
    inputs = enci(inputs')';
    
    if i == 1
        enc_init = enci;
        dec_init = deci;
    else
        enc_init = stack(enc_init, enci);
        dec_init = stack(deci, dec_init);
    end
end


%% Stack the RBMs
net_init = stack(enc_init, dec_init);
net_init.trainParam.epochs = max_epochs;
net_init.trainParam.showWindow = visualize;
net_init.divideFcn = 'dividetrain';
net_init.plotFcns = {'plotperform'};
net_init.plotParams = {nnetParam}; % Dummy?

%% Start fine tuning
net = train(net_init, X', X');

%% Set outputs
% Get encoder
if nargout > 1
    enc = [];
    for i = 1:length(num_hidden)
        if i == 1, enc = get_layer(net, 1); else enc = stack(enc, get_layer(net, i)); end
    end
    varargout{1} = enc;
end

% Get decoder
if nargout > 2
    dec = [];
    for i = (length(num_hidden)+1):(2*length(num_hidden))
        if i == (length(num_hidden)+1), dec = get_layer(net, (length(num_hidden)+1)); else dec = stack(dec, get_layer(net, i)); end
    end
    varargout{2} = dec;
end

% Get initial encoder
if nargout > 3
    varargout{3} = enc_init;
end

% Get initial decoder
if nargout > 4
    varargout{4} = dec_init;
end    

end
