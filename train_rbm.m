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
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units, can be 'logsig', 'tansig' or 'purelin'
%
%       'VisibleFunction' ('logsig'): transfer function for the visible
%       units, can be 'logsig', 'tansig' or 'purelin'
%
%       'UnitFunction' ('default'): function determining the type of hidden
%       units, can be 'binary', 'gaussian' or 'default'. When using
%       'default', the type of units depends on the transfer function. For
%       'purelin' the default is 'gaussian', otherwise it is set to
%       'binary'.
%
%       'MaxEpochs' (50): number of training iterations
%
%       'NumBatches' (100): number of mini-batches considered in each epoch
%
%       'LearningRate' (0.1): learning rate
%
%       'Momentum' (0.9): momentum
%
%       'Regularizer' (0.0002): regularizer for the weight update
%
%       'Sigma' (0.1): standard deviation for the random Gaussian
%       distribution used for initializing the weights
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

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('HiddenFunction', 'logsig', @ischar)
p.addParameter('VisibleFunction', 'logsig', @ischar)
p.addParameter('UnitFunction', 'default', @ischar)
p.addParameter('MaxEpochs', 50, @isnumeric)
p.addParameter('NumBatches', 100, @isnumeric)
p.addParameter('LearningRate', 0.1, @isfloat)
p.addParameter('Momentum', 0.9, @isfloat)
p.addParameter('Regularizer', 0.0002, @isfloat)
p.addParameter('Sigma', 0.1, @isfloat)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.addParameter('Callback', 0);
p.parse(varargin{:});
% Get opts
hidden_function = p.Results.HiddenFunction;
visible_function = p.Results.VisibleFunction;
unit_function = p.Results.UnitFunction;
max_epochs = p.Results.MaxEpochs;
num_batches = p.Results.NumBatches;
regularizer = p.Results.Regularizer;
sigma = p.Results.Sigma;
learning_rate = p.Results.LearningRate;
momentum = p.Results.Momentum;
row_major = p.Results.RowMajor;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
callback = p.Results.Callback;
% Transpose data to ensure row-major
if ~row_major, X = X'; end
% Get unit function
if strcmpi(unit_function, 'default')
    unit_function = 'binary';
    if strcmpi(hidden_function, 'purelin')
        unit_function = 'gaussian';
    end
end
% Check transfer/unit functions
assert(exist(hidden_function) > 0, 'Unknown hidden transfer function: %s!\n', hidden_function);
assert(exist(visible_function) > 0, 'Unknown visible transfer function: %s!\n', visible_function);
assert(exist(unit_function) > 0, 'Unknown unit function: %s!\n', unit_function);

%% Initialize weights and biases and their increments
[N, num_visible] = size(X);
wh = sqrt(num_visible);
W = sigma * randn(num_visible, num_hidden);
Bvis = zeros(1, num_visible);
Bhid = zeros(1, num_hidden);
Winc = zeros(size(W));
Bvisinc = zeros(size(Bvis));
Bhidinc = zeros(size(Bhid));

%% Prepare other stuff
if visualize
    figname = 'RBM';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    figure('Name', figname)
    % If image data
    if wh == round(wh)
        h1 = subplot(221);
        h2 = subplot(222);
        h3 = subplot(223);
        h4 = subplot(224);
    else
        h1 = gca;
    end
end

%% Setup mini-batches
Nbatch = N;
if num_batches > 1, Nbatch = floor(N / num_batches); end

%% Verbosity
if verbose
    fprintf('****************************************************************************\n');
    fprintf('Training a %i-%i RBM using %i training examples and a batch size of %i\n', num_visible, num_hidden, N, Nbatch);
    fprintf('Using hidden and visible unit transfer functions ''%s'' and ''%s''\n', hidden_function, visible_function);
    fprintf('Using unit function ''%s''\n', unit_function);
    fprintf('****************************************************************************\n');
end

%% Train
perf = zeros(1,max_epochs);
for epoch = 1:max_epochs
    % Verbosity
    if verbose
        tic
        fprintf('********** Epoch %i/%i **********\n', epoch, max_epochs);
    end
    
    % Loop over batches
    err = 0;
    chars = 0;
    for batch_begin = 1:Nbatch:N
        %% Initialize batch data
        batch_end = min([batch_begin + Nbatch - 1, N]);
        batch_size = batch_end - batch_begin + 1;
        
        % Verbosity
        if verbose
            for i = 1:chars, fprintf('\b'); end
            chars = fprintf('%i/%i', batch_end, N);
            if batch_end == N, fprintf('\n'); end
        end
        
        % Get batch data
        if batch_end > N
            Xb = X(batch_begin:end,:);
        else
            Xb = X(batch_begin:batch_end,:);
        end
        
        %% Positive phase
        % Forward pass through first layer
        pos_hidden_activations = feval(hidden_function, Xb * W + repmat(Bhid, batch_size, 1));
        % Apply unit function
        pos_hidden_states = feval(unit_function, pos_hidden_activations, batch_size, num_hidden);
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
    perf(epoch) = err/numel(X);
    
    % Verbosity
    if verbose
        fprintf('Error (SSE/MSE): %.0f/%.5f\n', err, perf(epoch));
        fprintf('Computation time [s]: %.5f\n', toc);
    end
    
    % Visualization
    if visualize
        % Plot performance
        plot(h1, 1:epoch, perf(1:epoch), '-*k', 'LineWidth', 1.5)
        xlim(h1, [0.9 epoch+1.1])
        ylim(h1, [0 1.1*max(perf)])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance (MSE)')
        
        % If image data
        if round(wh) == wh
            % Show first image
            imshow(reshape(Xb(1,:)', [wh wh]), 'parent', h2)
            title(h2, 'Image')
            
            % Show reconstruction
            imshow(reshape(neg_output_activations(1,:)', [wh wh]), 'parent', h3)
            title(h3, 'Reconstruction')

            % Show first neuron
            imagesc(reshape(W(:,1), [wh wh]), 'parent', h4)
            colorbar
            title(h4, 'First unit')
        end
        
        % Update figures
        colormap gray
        drawnow
    end % End visualization
    
    % Finally invoke the callback
    if isa(callback, 'function_handle'), feval(callback, epoch); end
end % End loop over epochs

%% Create output
enc = create_layer(num_visible, num_hidden, hidden_function, W', Bhid', 'traincgp', 'Name', 'Encoder');
dec = create_layer(num_hidden, num_visible, visible_function, W, Bvis', 'traincgp', 'Name', 'Decoder');

end

%% Unit functions
% Binary
function states = binary(activations, N, num_hidden)
states = activations > rand(N, num_hidden);
end

% Gaussian
function states = gaussian(activations, N, num_hidden)
states = activations + randn(N, num_hidden);
end