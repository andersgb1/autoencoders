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
%       'Sigma' (0.1): standard deviation for the random Gaussian
%       distribution used for initializing the weights
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
p.addParameter('LearningRate', 0.05, @isfloat)
p.addParameter('LearningRateMul', 1, @isfloat)
p.addParameter('Momentum', 0.9, @isfloat)
p.addParameter('MomentumInc', 0, @isfloat)
p.addParameter('Regularizer', 0.0005, @isfloat)
p.addParameter('Sigma', 0.1, @isfloat)
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
linact = {'purelin', 'poslin', 'satlin'};
linear_inputs = any(strcmpi(input_function, linact));

% Check dimensions
if width > 0 % Image data
    assert(round(width) == width, 'Specified width is non-integer!')
    height = size(X,2) / width;
    assert(round(height) == height, 'Invalid width!')
elseif round(sqrt(size(X,2))) == sqrt(size(X,2)) % Quadratic dimension, can also be shown
    width = sqrt(size(X,2));
    height = width;
end

% In case of linear visible units, standardize the input
if linear_inputs
    warning('Linear input units selected! Standardizing the dataset...');
    X = (X - repmat(mean(X), size(X,1), 1)) / std(X(:));
elseif strcmpi(input_function, 'logsig')
    warning('Logistic input units selected! Normalizing dataset to [0,1]...');
    X = (X - min(X(:))) / (max(X(:)) - min(X(:)));
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
        if i == 1 % First AE
            encfun = input_function;
            decfun = hidden_function;
            w = width;
        elseif i == length(num_hidden) % Final AE
            encfun = hidden_function;
            decfun = output_function;
        else
            encfun = hidden_function;
            decfun = hidden_function;
        end
        
        % TODO: Reduce larning rate for linear activation functions
        if any( [strcmpi(encfun, linact) strcmpi(decfun,  linact)] )
            learnrate = learning_rate / 100;
            sig = sigma / 10;
            warning(['Linear input and/or hidden units for AE %i selected!\n'...
            '\tScaling learning rate: %.0e --> %.0e\n'...
            '\tScaling initial weight power: %.2f --> %.2f'],...
            i, learning_rate, learnrate, sigma, sig);
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
    end
end

% net = train(net, X', X');
net = sgd(net, X', X', max_epochs, learning_rate,...
    'Batches', batches,...
    'ValidationFraction', val_frac,...
    'Loss', loss,...
    'Adadelta', true,...
    'Momentum', momentum,...
    'Regularizer', regularizer,...
    'Verbose', verbose,...
    'Visualize', visualize,...
    'CheckpointFile', 'sgd_sae.mat');

if resume, save(netfile, 'net', 'net_init'); end

if nargout > 1, varargout{1} = net_init; end

return

%% Setup mini-batches
N = size(X,1);
if isempty(batches), batches = {1:N}; end
batches_train = batches;

Nbatch = length(batches);
Nval = 0;
if val_frac > 0
    Nval = round(val_frac * Nbatch);
    if Nval > 0
        Nbatch = Nbatch - Nval;
        batches_val = batches{(Nbatch+1):(Nbatch+Nval)}; % Produces a vector
        batches_train = batches(1:Nbatch); % Produces a cell array
        Xval = X(batches_val,:);
        perf_val = zeros(1, max_epochs);
    end
end

%% Resume
iter_start = 1;
% historical_grad = 1;
if resume && exist(netfile, 'file')
    load(netfile);
    iter_start = iter;
    
    % Update learning rate and momentum
    for i = 1:(iter-1)
        learning_rate = learning_rate * learning_rate_mul;
        momentum = momentum + momentum_inc;
    end
end

%% Prepare viz
if iter_start <= max_epochs && visualize
    figname = 'SAE';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    hfig = figure('Name', figname);
    % If image data
    if width > 0
        h1 = subplot(2,2,1);
        h2 = subplot(2,2,2);
        h3 = subplot(2,2,3);
        h4 = subplot(2,2,4);
    else
        h1 = subtightplot(1,2,1);
        h2 = subtightplot(1,2,2);
    end
end

%% Verbosity
if verbose
    if iter_start < max_epochs
        fprintf('****************************************************************************\n');
        fprintf('Fine tuning the SAE for %i epochs using %i training examples\n', max_epochs, N);
        if iter_start > 1, fprintf('Resuming from epoch %i\n', iter_start); end
        if Nval > 0
            fprintf('Using %i/%i batches for training/validation\n', Nbatch, Nval);
        else
            fprintf('Using %i training batches\n', Nbatch);
        end
        fprintf('****************************************************************************\n');
    else
        fprintf('Fine tuned network already exists! Skipping fine tuning...\n');
    end
end

%% Start backpropagation
% grad = zeros(1, max_epochs);
% fudge_factor = 1e-6;
lr_dec = 0; % Number of times we decreased the learning rates
perf = zeros(1, max_epochs);
Winc = 0;
for epoch = iter_start:max_epochs
    % Verbosity
    if verbose, fprintf('********** Epoch %i/%i (lr: %.0e, mom: %.2f, reg: %.0e) ********** \n', epoch, max_epochs, learning_rate, momentum, regularizer); end
    
    % Shuffle X
    order = randperm(size(X,1));
    X = X(order,:);
    
    % Loop over batches
    lossbatch = zeros(1, Nbatch);
    for j=1:Nbatch
        % Verbosity
        if verbose, chars = fprintf('\tBatch %i/%i\n', j, Nbatch); end
        
        % Get batch data
        Xb = X(batches_train{j},:);
        
        % Get current weights
        w = getwb(net);
        
        % Run minimization
        % SCG
%         options = zeros(1,18);
%         options(1) = -1; % Display SCG progress?
%         options(14) = 3; % Maximum SCG iterations
%         [w, ~, ~] = scg(@f, w', options, @df, net, Xb');
        
        % Momentum and learning rate
%         gradj = df(w, net, Xb')';
%         Winc = momentum * Winc - learning_rate * gradj - learning_rate * regularizer * w;
%         w = w + Winc;
        
        gradj = -backprop(net, Xb', Xb', 'Loss', loss);
        Winc = momentum * Winc - learning_rate * gradj - learning_rate * regularizer * w;
        w = w + Winc;
        
%         % AdaGrad
%         gradj = df(w, net, Xb')';
%         historical_grad = historical_grad + gradj.^2;
%         adjusted_grad = gradj ./ (fudge_factor + sqrt(historical_grad));
%         Winc = -adjusted_grad - regularizer * w;
% %         Winc = -learning_rate * adjusted_grad - regularizer * w;
%         w = w + Winc;
        
%         % Compute error of mini-batch
%         perfbatch(j) = flog(end); % MSE for current batch
%         perfbatch(j) = f(w, net, Xb');
%         gradbatch(j) = norm(gradj);
        
        % Update weights
        net = setwb(net, w);
        
        % Compute error of mini-batch
%         lossbatch(j) = floss(Xb', net(Xb'));
        lossbatch(j) = backprop_loss(Xb', net(Xb'), loss);
        
        % Verbosity
        if verbose
            chars = chars + fprintf('\tReconstruction error of mini-batch: %f\n', lossbatch(j));
        end
        
        % Visualization
        if visualize
            % h1 is shown below
            
            % Plot performance of current mini-batch
            plot(h2, 1:j, lossbatch(1:j), '-k')
            xlim(h2, [0.5 j+0.5])
            if j > 1, set(h2, 'xtick', [1 j]); end
            ylim(h2, [0 inf])
            xlabel(h2, 'Mini-batch');
            ylabel(h2, 'Training performance ')
            
            % If image data
            if width > 0
                Xnet = net(X(1,:)');
                immin = min(min(Xnet), min(X(1,:)));
                immax = max(max(Xnet), max(X(1,:)));
                
                % Show first image
                imshow(reshape(X(1,:)', [height width]), [immin immax], 'parent', h3)
                colorbar(h3)
                title(h3, 'Image')
                axis(h3, 'off')
                
                % Show reconstruction
                imshow(reshape(Xnet, [height width]), [immin immax], 'parent', h4)
                colorbar(h4)
                title(h4, 'Reconstruction')
                axis(h4, 'off')
            end
            
            % Update figures
            colormap(hfig, 'gray');
            drawnow
        end % End visualization
    end % End loop over batches
    
    % Update learning rate and momentum
    learning_rate = learning_rate * learning_rate_mul;
    momentum = momentum + momentum_inc;
    
    % Store performance
    perf(epoch) = sum(lossbatch) / Nbatch;
%     if Nval > 0, perf_val(epoch) = floss(Xval', net(Xval')); end
    if Nval > 0, perf_val(epoch) = backprop_loss(Xval', net(Xval'), loss); end
    
    % Verbosity
    if verbose
        if Nval > 0
            fprintf('Training/validation error: %f/%f\n', perf(epoch), perf_val(epoch));
        else
            fprintf('Training error: %f\n', perf(epoch));
        end
        fprintf('******************************\n');
    end
    
    % Visualization
    if visualize
        % Plot performance of current epoch
        plot(h1, 1:epoch, perf(1:epoch), '-*k', 'LineWidth', 1.5)
        if Nval > 0
            hold(h1, 'on')
            plot(h1, 1:epoch, perf_val(1:epoch), '-r', 'LineWidth', 1.5)
            legend(h1, 'Training', 'Validation', 'Location', 'best')
            hold(h1, 'off')
        end
        xlim(h1, [0.5 epoch+0.5])
        if epoch > 1, set(h1, 'xtick', [1 epoch]); end
        ylim(h1, [0 inf])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance')
        if Nval > 0
            title(h1, sprintf('Best train/val performance: %f/%f', min(perf(1:epoch)), min(perf_val(1:epoch))))
        else
            title(h1, sprintf('Best performance: %f', min(perf(1:epoch))))
        end
            
        % Update figures
        colormap(hfig, 'gray');
        drawnow
    end
    
    % Save state
    if resume
        if verbose, fprintf('Saving fine tuned net for epoch %i...\n', epoch); end
        iter = epoch + 1; % Save instead the next epoch index
        if Nval > 0
            save(netfile, 'net', 'net_init', 'iter', 'Winc', 'perf', 'perf_val');
        else
            save(netfile, 'net', 'net_init', 'iter', 'Winc', 'perf');
        end
    end
    
    % Termination
    if Nval > 0 && epoch > 1
        if perf_val(epoch) >= perf_val(epoch-1)
            fprintf('Validation error has stagnated at %f!', perf_val(epoch));
            if lr_dec < 3
                tmp = learning_rate / 10;
                fprintf('\tScaling learning rate: %f --> %f...\n', learning_rate, tmp);
                learning_rate = tmp;
                lr_dec = lr_dec + 1;
            else
                fprintf('\tStopping backpropagation...\n');
                break
            end
        end
    end
end % End epochs

if visualize && exist('hfig') > 0, print(hfig, figname, '-dpdf'); end

%% Set outputs
if nargout > 1, varargout{1} = net_init; end

end

%% Objective function
function err = f(w, net, X)
wnet = setwb(net, w);
err = perform(wnet, X, wnet(X));
end

%% Derivative of the objective function wrt. w
function grad = df(w, net, X)
wnet = setwb(net, w);
grad = -defaultderiv('dperf_dwb', wnet, X, X)';
end
