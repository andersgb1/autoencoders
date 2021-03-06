function [enc,dec] = train_rbm(X, num_hidden, varargin)
% TRAIN_RBM  Train a restricted Boltzmann machine
%   [enc,dec] = TRAIN_RBM(X, num_hidden, ...) trains an RBM on the data X
%   using a hidden layer of size num_hidden. The result is returned as a
%   pair where enc is the encoder part and dec is the decoder. These can be
%   stacked in an RBM using the MATLAB function STACK.
%
%   The data in X is expected to be row-major, meaning that all the feature
%   vectors are stacked as rows on top of each other. If your data is in
%   column-major, use the 'RowMajor' option described below.
%
%   Name value pair options (default value):
%
%       'VisibleFunction' ('logsig'): transfer function for the visible
%       units, can be 'logsig', 'tansig' 'purelin', etc.
%
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units
%
%       'SamplingFunction' ('default'): function determining the type of
%       hidden units, can be 'binary', 'gaussian' or 'default'. When using
%       'default', the type of units depends on the transfer function. For
%       linear functions ('purelin', 'poslin', etc.) the default is
%       'gaussian', otherwise it is set to 'binary'.
%
%       'MaxEpochs' (50): number of training iterations
%
%       'Batches' (empty cell): mini-batches considered in each epoch. If
%       you want to split the training data into mini-batches during each
%       epoch, this argument should contain a cell array, each element
%       being indices for a mini-batch.
%
%       'ValidationFraction' (0.1): set this to a value in [0,1[ to use a
%       fraction of the training data as validation set during
%       training. This also has the consequence that training
%       will be terminated as soon as the validation error stagnates.
%
%       'LearningRate' (0.1): learning rate
%
%       'Momentum' (0.9): momentum
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
%       that all observations in X have a 2D structure and can be
%       visualized as an image with this width
%
%       'Verbose' (false): logical, set to true to print status messages
%
%       'Visualize' (false): logical, set to true to show status plots
%
%       See also TRAIN_DBN.

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('VisibleFunction', 'logsig', @ischar)
p.addParameter('HiddenFunction', 'logsig', @ischar)
p.addParameter('SamplingFunction', 'default', @ischar)
p.addParameter('MaxEpochs', 50, @isnumeric)
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('LearningRate', 0.1, @isfloat)
p.addParameter('Momentum', 0.9, @isfloat)
p.addParameter('Regularizer', 0.0005, @isfloat)
p.addParameter('Sigma', 0.1, @isfloat)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Width', 0, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.parse(varargin{:});
% Get opts
visible_function = p.Results.VisibleFunction;
hidden_function = p.Results.HiddenFunction;
sampling_function = p.Results.SamplingFunction;
max_epochs = p.Results.MaxEpochs;
batches = p.Results.Batches;
val_frac = p.Results.ValidationFraction;
assert(val_frac >= 0 && val_frac < 1, 'Validation fraction must be a number in [0,1[!')
regularizer = p.Results.Regularizer;
sigma = p.Results.Sigma;
learning_rate = p.Results.LearningRate;
momentum = p.Results.Momentum;
row_major = p.Results.RowMajor;
width = p.Results.Width;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
% Transpose data to ensure row-major
if ~row_major, X = X'; end
% Get unit function
funs = {'logsig', 'tanh', 'tansig'};
if strcmpi(sampling_function, 'default')
    sampling_function = 'gaussian';
    if any( strcmpi(hidden_function, funs) )
        sampling_function = 'binary';
    end
end
% Check transfer/unit functions
assert(any(strcmpi(hidden_function, funs)), 'Unknown hidden transfer function: %s!\n', hidden_function);
assert(any(strcmpi(visible_function, funs)) > 0, 'Unknown visible transfer function: %s!\n', visible_function);
assert(any(strcmpi(sampling_function, {'gaussian', 'binary'})) > 0, 'Unknown sampling function: %s!\n', sampling_function);

%% Initialize dimensions, weights and biases and their increments
[N, num_visible] = size(X);
if width > 0 % Image data
    assert(round(width) == width, 'Specified width is non-integer!')
    height = num_visible / width;
    assert(round(height) == height, 'Invalid width!')
elseif round(sqrt(num_visible)) == sqrt(num_visible) % Quadratic dimension, can also be shown
    width = sqrt(num_visible);
    height = width;
end
W = sigma * randn(num_visible, num_hidden);
Bvis = zeros(1, num_visible);
Bhid = zeros(1, num_hidden);
Winc = zeros(size(W));
Bvisinc = zeros(size(Bvis));
Bhidinc = zeros(size(Bhid));

%% Prepare other stuff
if visualize
    figname = sprintf('RBM %i-%i', num_visible, num_hidden);
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    hfig = figure('Name', figname);
    % If image data
    if width > 0
        h1 = subplot(131);
        h3 = subplot(132);
        h4 = subplot(133);
    else
        h1 = gca;
    end
end

%% Setup mini-batches
if isempty(batches), batches = {1:N}; end

Nbatch = length(batches);
Nval = 0;
if val_frac > 0
    Nval = round(val_frac * Nbatch);
    if Nval > 0
        Nbatch = Nbatch - Nval;
        batches_val = batches{(Nbatch+1):(Nbatch+Nval)}; % Produces a vector
        batches = batches(1:Nbatch); % Produces a cell array
        Xval = X(batches_val,:);
        perf_val = zeros(1, max_epochs);
    end
end

%% Verbosity
if verbose
    fprintf('****************************************************************************\n');
    fprintf('Training a %i-%i RBM using %i training examples\n', num_visible, num_hidden, N);
        if Nval > 0
            fprintf('Using %i/%i batches for training/validation\n', Nbatch, Nval);
        else
            fprintf('Using %i training batches\n', Nbatch);
        end
    fprintf('Using hidden and visible unit transfer functions ''%s'' and ''%s''\n', hidden_function, visible_function);
    fprintf('Using sampling function ''%s''\n', sampling_function);
    fprintf('****************************************************************************\n');
end

%% Train
perf = zeros(1,max_epochs);
lr_dec = 0; % Number of times we decreased the learning rates
for epoch = 1:max_epochs
    % Verbosity
    if verbose
        tic
        fprintf('********** Epoch %i/%i **********\n', epoch, max_epochs);
    end
    
    % Shuffle X
    order = randperm(size(X,1));
    X = X(order,:);
    
    % Loop over batches
    err = 0;
    train_numel = 0;
    chars = 0;
    for i = 1:Nbatch
        %% Verbosity
        if verbose
            for j = 1:chars, fprintf('\b'); end
            chars = fprintf('Batch %i/%i of size %i (lr: %.0e, mom: %.2f, reg: %.0e)',...
                i, Nbatch, length(batches{i}),...
                learning_rate, momentum, regularizer);
            if i == Nbatch, fprintf('\n'); end
        end
        
        %% Get batch data
        Xb = X(batches{i},:);
        batch_size = size(Xb,1);
        train_numel = train_numel + numel(Xb);
        
        %% Positive phase
        % Forward pass through first layer
        pos_hidden_activations = feval(hidden_function, Xb * W + repmat(Bhid, batch_size, 1));
        
        % Apply sampling function
        pos_hidden_states = feval(sampling_function, pos_hidden_activations, batch_size, num_hidden);
        % Get the positive gradient
        pos_gradient = Xb' * pos_hidden_activations;
        
        %% Negative phase
        % Reconstruction
        neg_output_activations = feval(visible_function, pos_hidden_states * W' + repmat(Bvis, batch_size, 1));
        % Now use the reconstructed signal to resample hidden activations
        neg_hidden_activations = feval(hidden_function, neg_output_activations * W + repmat(Bhid, batch_size, 1));
        % Get the negative gradient
        neg_gradient = neg_output_activations' * neg_hidden_activations;

        %% Update weights and biases
        Winc = momentum * Winc + learning_rate * ( (pos_gradient - neg_gradient) / batch_size - regularizer * W );
        W = W + Winc;
        
        % Bias update for visible units
        pos_visible_activations = sum(Xb);
        neg_visible_activations = sum(neg_output_activations);
        Bvisinc = momentum * Bvisinc + learning_rate * (pos_visible_activations - neg_visible_activations) / batch_size;
        Bvis = Bvis + Bvisinc;
        
        % Bias update for hidden units
        pos_hiddden_activations = sum(pos_hidden_activations);
        neg_hidden_activations = sum(neg_hidden_activations);
        Bhidinc = momentum * Bhidinc + learning_rate * (pos_hiddden_activations - neg_hidden_activations) / batch_size;
        Bhid = Bhid + Bhidinc;

        %% Compute error
        err = err + sse(Xb - neg_output_activations);
    end % End loop over batches
    
    % Store performance
    perf(epoch) = err / train_numel;
    if Nval > 0
        Nvalcases = size(Xval, 1);
        hidact = feval(hidden_function, Xval * W + repmat(Bhid, Nvalcases, 1));
        visact = feval(visible_function, hidact * W' + repmat(Bvis, Nvalcases, 1));
        perf_val(epoch) = mse(Xval' - visact');
    end
    
    % Verbosity
    if verbose
        if Nval > 0
            fprintf('Training/validation error: %f/%f\n', perf(epoch), perf_val(epoch));
        else
            fprintf('Training error: %\n', perf(epoch));
        end
        fprintf('Computation time [s]: %.2f\n', toc);
        fprintf('******************************\n');
    end
    
    % Visualization
    if visualize
        % Plot performance
        plot(h1, 1:epoch, perf(1:epoch), '-*k', 'LineWidth', 1.5)
        if Nval > 0
            hold(h1, 'on')
            plot(h1, 1:epoch, perf_val(1:epoch), '-r', 'LineWidth', 1.5)
            legend(h1, 'Training', 'Validation', 'Location', 'best')
            hold(h1, 'off')
        end
        xlim(h1, [0.9 epoch+1.1])
        if epoch > 1, set(h1, 'xtick', [1 epoch]); end
        ylim(h1, [0 1.1*max(perf)])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance (MSE)')
        
        % If image data
        if width > 0
            % Show first image
            imagesc(reshape(Xb(1,:)', [height width]), 'parent', h3)
            colorbar(h3)
            title(h3, 'Image')
            axis(h3, 'equal', 'off')
            
            % Show reconstruction
            imagesc(reshape(neg_output_activations(1,:)', [height width]), 'parent', h4)
            colorbar(h4)
            title(h4, 'Reconstruction')
            axis(h4, 'equal', 'off')
            
            colormap gray
            
            % Show the strongest neurons
            plot_neurons(W, width);
        end
        
        % Update figures
        drawnow
    end % End visualization
    
%     % Termination
%     if epoch > 1 && perf(epoch) >= perf(epoch-1)
%         fprintf('Training error has stagnated at %f! Stopping pretraining...\n', perf(epoch))
%         break;
%     end
    if Nval > 0 && epoch > 1
        if perf_val(epoch) >= perf_val(epoch-1)
            fprintf('Validation error has stagnated at %f\n!', perf_val(epoch));
            if lr_dec < 3
                fprintf('\tDiscarding weight change with norm %.0e\n', norm(Winc(:)));
                perf(epoch) = perf(epoch-1);
                perf_val(epoch) = perf_val(epoch-1);
                W = W - Winc;
                tmp = learning_rate / 10;
                fprintf('\tScaling learning rate: %.0e --> %.0e...\n', learning_rate, tmp);
                learning_rate = tmp;
                lr_dec = lr_dec + 1;
            else
                fprintf('\tStopping pretraining...\n');
                break
            end
        end
    end
end % End loop over epochs

if visualize, print(hfig, figname, '-dpdf'); end

%% Create output
enc = create_layer(num_visible, num_hidden, hidden_function, W', Bhid', 'traincgp', 'Name', 'Encoder');
dec = create_layer(num_hidden, num_visible, visible_function, W, Bvis', 'traincgp', 'Name', 'Decoder');

end

%% Unit functions
% Binary
function states = binary(activations, N, num_hidden)
states = double(activations > rand(N, num_hidden));
end

% Gaussian
function states = gaussian(activations, N, num_hidden)
states = activations + randn(N, num_hidden);
end