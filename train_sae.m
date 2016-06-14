function [net,varargout] = train_sae(X, num_hidden, varargin)
% TRAIN_SAE  Train a stack of (denoising) autoencoders
%   net = TRAIN_SAE(X, num_hidden, ...) trains an SAE on the data X using a
%   stack of (denoising) autoencoders. The sizes of the hidden layers
%   is given in the vector num_hidden. The result is returned as a MATLAB
%   network.
%
%   [net,enc] = TRAIN_SAE(...) additionally returns the encoder part of the
%   network.
%
%   [net,enc,dec] = TRAIN_SAE(...) additionally returns the decoder part of
%   the network.
%
%   [net,enc,dec,enc_init] = TRAIN_SAE(...) additionally returns the
%   encoder of the initialized network before fine tuning
%
%   [net,enc,dec,enc_init,dec_init] = TRAIN_SAE(...) additionally returns
%   the decoder of the initialized network before fine tuning
%
%   The data in X is expected to be row-major, meaning that all the feature
%   vectors are stacked as rows on top of each other. If your data is in
%   column-major, use the 'RowMajor' option described below.
%
%   Name value pair options (default value):
%
%       'InputFunction' ('logsig'): transfer function applied to the
%       visible units, can be 'logsig', 'tansig' 'purelin', 'poslin' and
%       'satlin'
%
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units
%
%       'OutputFunction' ('logsig'): transfer function for the hidden
%       units in the final layer
%
%       'TiedWeights' (true): use tied (symmetric) weights for the
%       encoder/decoder
%
%       'MaxEpochsInit' (50): number of training iterations for the
%       layer-wise initialization. If a single number is given, it applies
%       to all layers, otherwise you can specify a vector giving a number
%       for each layer.
%
%       'MaxEpochs' (200): number of training iterations for the fine
%       tuning based on backpropagation
%
%       'Loss' ('mse'): loss function for the output, can be
%       'crossentropy', 'log' (equal to 'crossentropy'), 'mse', 'mae',
%       'crossentropy_binary' and 'binary_crossentropy'
%
%       'BatchesInit' (empty cell): mini-batches considered in each epoch
%       of pretraining. If you want to split the training data into
%       mini-batches during each epoch, this argument should contain a cell
%       array, each element being indices for a mini-batch.
%
%       'Batches' (empty cell): mini-batches considered in each epoch
%       of backpropagation.
%
%       'ValidationFraction' (0.1): set this to a value in [0,1[ to use a
%       fraction of the training data as validation set during
%       training. This also has the consequence that training
%       will be terminated as soon as the validation error stagnates.
%
%       'MaskingNoise' (0): turn the autoencoders into denoising
%       autoencoders by introducing masking noise (randomly setting inputs
%       to zero) in the interval [0,1[ - this only applies to pretraining
%
%       'GaussianNoise' (0): turn the autoencoder into a denoising
%       autoencoder by introducing Gaussian noise with a standard deviation
%       as provided
%
%       'DropoutRate' (0): set this to a number in ]0,1[ for dropout
%       in the hidden layer
%
%       'LearningRate' (0.1): learning rate for both pretraining and fine
%       tuning. If learning rate decay is non-zero this indicates the start
%       learning rate for fine tuning. NOTE: For pretraining, the learning
%       rate is scaled by 1/100 for linear units!
%
%       'LearningRateMul' (1): multiplicative learning rate decay
%
%       'Momentum' (0.5): momentum, if momentum increase is non-zero this
%       indicates the start momentum
%
%       'MomentumInc' (0): additive momentum increase
%
%       'Regularizer' (0.0005): regularizer for the weight update
%
%       'Sigma' (0): standard deviation for the random Gaussian
%       distribution used for initializing the weights - zero means
%       automatic
%
%       'RowMajor' (true): logical specifying whether the observations in X
%       are placed in rows or columns
%
%       'Width' (0): if set to a positive integer value, this indicates
%       that all observations in X have a rectangular 2D structure and can
%       be visualized as an image with this width - for quadratic
%       dimensions in X, this is automatically detected
%
%       'Verbose' (false): logical, set to true to print status messages
%
%       'Visualize' (false): logical, set to true to show status plots
%
%       'UseGPU' (false): set to true to use GPU if available - note that
%       this demotes all datatypes to single precision floats
%
%       'Resume' (false): logical, if set to true, allow for resuming the
%       pretraining. This means that during the
%
%       See also TRAIN_DBN.

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('InputFunction', 'logsig', @ischar)
p.addParameter('HiddenFunction', 'logsig', @ischar)
p.addParameter('OutputFunction', 'logsig', @ischar)
p.addParameter('TiedWeights', true, @islogical)
p.addParameter('MaxEpochsInit', 50, @isnumeric)
p.addParameter('MaxEpochs', 200, @isnumeric)
p.addParameter('Loss', 'mse', @ischar)
p.addParameter('BatchesInit', {}, @iscell)
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('MaskingNoise', 0, @isnumeric)
p.addParameter('GaussianNoise', 0, @isnumeric)
p.addParameter('DropoutRate', 0, @isnumeric)
p.addParameter('LearningRate', 0.05, @isnumeric)
p.addParameter('LearningRateMul', 1, @isnumeric)
p.addParameter('Momentum', 0.9, @isnumeric)
p.addParameter('MomentumInc', 0, @isnumeric)
p.addParameter('Regularizer', 0.0005, @isnumeric)
p.addParameter('Sigma', 0, @isnumeric)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Width', 0, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.addParameter('UseGPU', false, @islogical)
p.addParameter('Resume', false, @islogical)
p.parse(varargin{:});
% Get opts
input_function = p.Results.InputFunction;
hidden_function = p.Results.HiddenFunction;
output_function = p.Results.OutputFunction;
tied_weights = p.Results.TiedWeights;
max_epochs_init = p.Results.MaxEpochsInit;
if length(max_epochs_init) == 1
    max_epochs_init = max_epochs_init * ones(1, length(num_hidden));
end
assert(length(max_epochs_init) == length(num_hidden), 'You must specify as many initial epochs as layers!')
max_epochs = p.Results.MaxEpochs;
loss = p.Results.Loss;
batches_init = p.Results.BatchesInit;
batches = p.Results.Batches;
val_frac = p.Results.ValidationFraction;
assert(val_frac >= 0 && val_frac < 1, 'Validation fraction must be a number in [0,1[!')
mask_noise = p.Results.MaskingNoise;
assert(mask_noise >= 0 && mask_noise < 1, 'Masking noise level must be a number in [0,1[!')
gauss_noise = p.Results.GaussianNoise;
dropout = p.Results.DropoutRate;
regularizer = p.Results.Regularizer;
sigma = p.Results.Sigma;
learning_rate = p.Results.LearningRate;
learning_rate_mul = p.Results.LearningRateMul;
momentum = p.Results.Momentum;
momentum_inc = p.Results.MomentumInc;
row_major = p.Results.RowMajor;
width = p.Results.Width;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
use_gpu = p.Results.UseGPU;
resume = p.Results.Resume;

% Transpose data to ensure row-major
if ~row_major, X = X'; end
% Get unit function
funs = {'logsig', 'tansig', 'purelin', 'poslin', 'satlin'};
assert(any(strcmpi(input_function, funs)) > 0, 'Unknown input transfer function: %s!\n', input_function);
assert(any(strcmpi(hidden_function, funs)), 'Unknown hidden transfer function: %s!\n', hidden_function);
assert(any(strcmpi(output_function, funs)) > 0, 'Unknown output transfer function: %s!\n', output_function);
if any(strcmpi(loss,  {'crossentropy_binary', 'binary_crossentropy', 'crossentropy', 'log'}))
    assert(strcmpi(output_function, 'logsig'), 'Cross-entropy losses only supported by logistic sigmoid output units!')
end


% Check dimensions
if width > 0 % Image data
    assert(round(width) == width, 'Specified width is non-integer!')
    height = size(X,2) / width;
    assert(round(height) == height, 'Invalid width!')
elseif round(sqrt(size(X,2))) == sqrt(size(X,2)) % Quadratic dimension, can also be shown
    width = sqrt(size(X,2));
    height = width;
end

% TODO: Standardize the dataset depending on the units of the first layer
if strcmpi(input_function, 'purelin')
    warning('Linear input units selected! Mean subtracting the dataset...');
    X = bsxfun(@minus, X, mean(X));
elseif any(strcmpi(input_function, {'logsig', 'satlin'}))
    warning('Logistic sigmoid or saturated linear input units selected! Normalizing dataset to [0,1]...');
    X = (X - min(X(:))) / (max(X(:)) - min(X(:)));
elseif any(strcmpi(input_function, {'tansig', 'satlins'}))
    warning('Tangent sigmoid or symmetric saturated linear input units selected! Normalizing dataset to [-1,1]...');
    X = 2 * (X - min(X(:))) / (max(X(:)) - min(X(:))) - 1;
end

% Set checkpoint
netfile = 'net.mat';

%% Start pretraining
if resume && exist(netfile, 'file')
    if verbose, fprintf('Pretrained network already exists! Skipping pretraining...\n'); end
    load(netfile);
else
    inputs = X;
    enc_init = [];
    dec_init = [];
    for i = 1:length(num_hidden)
        numhid = num_hidden(i);
        learnrate = learning_rate;
        sig = sigma;
        w = 0;
        if length(num_hidden) == 1 % First and only AE
            encfun = input_function;
            decfun = output_function;
            w = width;
        else
            if i == 1 % First of many AEs
                encfun = input_function;
                decfun = hidden_function;
                w = width;
            elseif i < length(num_hidden) % Intermediate AEs
                encfun = hidden_function;
                decfun = hidden_function;
            else % Final AE
                encfun = hidden_function;
                decfun = output_function;
            end
        end
        
        aefile = sprintf('ae%i.mat', i);
        if resume && exist(aefile, 'file')
            if verbose, fprintf('Loading AE %i from file...\n', i); end
            load(aefile);
        else
            [enci,deci] = train_ae(inputs, numhid,...
                'EncoderFunction', encfun,...
                'DecoderFunction', decfun,...
                'TiedWeights', tied_weights,...
                'MaxEpochs', max_epochs_init(i),...
                'Loss', loss,...
                'Batches', batches_init,...
                'ValidationFraction', val_frac,...
                'MaskingNoise', mask_noise,...
                'GaussianNoise', gauss_noise,...
                'DropoutRate', dropout,...
                'LearningRate', learnrate,...
                'Momentum', momentum,...
                'Regularizer', regularizer,...
                'Sigma', sig,...
                'Width', w,...
                'Verbose', verbose,...
                'Visualize', visualize,...
                'UseGPU', use_gpu);
            if resume, save(aefile, 'enci', 'deci'); end
        end
        
        % TODO: Capture error conditions here
        inputs = enci(inputs')';
        
        if i == 1
            enc_init = enci;
            dec_init = deci;
        else
            enc_init = stack(enc_init, enci);
            dec_init = stack(deci, dec_init);
        end
    end
    
    %% Stack the AEs
    net_init = stack(enc_init, dec_init);
    %     net_init.divideFcn = 'dividetrain';
    % TODO
%     net_init.performFcn = loss;
    net_init.performParam.regularization = regularizer;
    %     net_init.performParam.normalization = 'none';
    %     net_init.plotFcns = {'plotperform'};
    %     net_init.plotParams = {nnetParam}; % Dummy?
%         net_init.trainFcn = train_fcn;
    net_init.trainParam.epochs = max_epochs;
    %     net_init.trainParam.showWindow = visualize;
    %     net_init.trainParam.showCommandLine = verbose;
    %     net_init.trainParam.show = 1;
    net = net_init;
    
    % Save network
    if resume
        iter = 1;
        save(netfile, 'net', 'net_init', 'iter');
        for i=1:length(num_hidden), delete(sprintf('ae%i.mat', i)); end
    end
end

if max_epochs > 0
    net = sgd(net, X', X', max_epochs, learning_rate,...
        'Batches', batches,...
        'ValidationFraction', val_frac,...
        'Loss', loss,...
        'Optimizer', 'Adam',...
        'Momentum', momentum,...
        'Regularizer', regularizer,...
        'Verbose', verbose,...
        'Visualize', visualize,...
        'CheckpointFile', 'sgd_sae.mat');
    delete 'sgd_sae.mat';
end

if resume, save(netfile, 'net', 'net_init'); end

if nargout > 1, varargout{1} = net_init; end

return
