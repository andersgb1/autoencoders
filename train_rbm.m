function [enc,dec] = train_rbm(X, num_hidden, varargin)
% TRAIN_RBM  Train a restricted Boltzmann machine
%   rbm = TRAIN_RBM(X, num_hidden, ...) trains an RBM on the data X using a
%   hidden layer of size num_hidden.
%
%   Name value pair options (default value):
%
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units, can be 'logsig' or 'purelin'
%
%       'VisibleFunction' ('logsig'): transfer function for the visible
%       units, can be 'logsig' or 'purelin'
%
%       'UnitFunction' ('default'): function determining the types of
%       units, can be 'binary', 'gaussian' or 'default'. When using
%       'default', the type of units depends on the transfer function. For
%       'logsig', the default unit is 'binary' and for 'purelin' the
%       default is 'gaussian'.
%
%       'MaxEpochs' (50): number of training iterations
%
%       'NumBatches' (100): number of mini-batches
%
%       'LearningRate' (0.1): learning rate
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
p.addParameter('Regularizer', 0.0002, @isfloat)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.parse(varargin{:});
% Get opts
hidden_function = p.Results.HiddenFunction;
visible_function = p.Results.VisibleFunction;
unit_function = p.Results.UnitFunction;
max_epochs = p.Results.MaxEpochs;
num_batches = p.Results.NumBatches;
regularizer = p.Results.Regularizer;
learning_rate = p.Results.LearningRate;
row_major = p.Results.RowMajor;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
% Transpose data to ensure row-major
if ~row_major, X = X'; end
% Get hidden transfer function handle
if strcmpi(hidden_function, 'logsig')
    hidden_function_handle = @logsig;
elseif strcmpi(hidden_function, 'purelin')
    hidden_function_handle = @purelin;
else
    error('Unknown hidden transfer function: %s!\n', hidden_function);
end
% Get visible transfer function handle
if strcmpi(visible_function, 'logsig')
    visible_function_handle = @logsig;
elseif strcmpi(visible_function, 'purelin')
    visible_function_handle = @purelin;
else
    error('Unknown visible transfer function: %s!\n', visible_function);
end
% Get unit function handle
if strcmpi(unit_function, 'default')
    unit_function = 'binary';
    if strcmpi(hidden_function, 'purelin')
        unit_function = 'gaussian';
    end
end
if strcmpi(unit_function, 'binary')
    unit_function_handle = @binary;
elseif strcmpi(unit_function, 'gaussian')
    unit_function_handle = @gaussian;
else
    error('Unknown unit function: %s!\n', unit_function);
end

%% Initialize weights and biases
[N, num_visible] = size(X);
W = 0.1 * randn(num_visible, num_hidden);
Bvis = zeros(1, num_visible);
Bhid = zeros(1, num_hidden);

%% Prepare other stuff
if visualize
    figname = 'RBM';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    figure('Name', figname)
    h1 = subplot(221);
    h2 = subplot(222);
    h3 = subplot(223);
    h4 = subplot(224);
end

%% Setup mini-batches
Nbatch = 1;
if num_batches > 1, Nbatch = floor(N / num_batches); end

%% Verbosity
fprintf('****************************************************************************\n');
fprintf('Training a %i-%i RBM using %i training examples and a batch size of %i\n', num_visible, num_hidden, N, Nbatch);
fprintf('Using hidden and visible unit transfer functions ''%s'' and ''%s''\n', hidden_function, visible_function);
fprintf('Using unit function ''%s''\n', unit_function);
fprintf('****************************************************************************\n');

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
        pos_hidden_activations = hidden_function_handle(Xb * W + repmat(Bhid, batch_size, 1));
        % Apply unit function
        pos_hidden_states = unit_function_handle(pos_hidden_activations, batch_size, num_hidden);
        % Get the positive gradient
        pos_gradient = Xb' * pos_hidden_activations;
        
        %% Negative phase
        % Reconstruction
        neg_output_activations = visible_function_handle(pos_hidden_states * W' + repmat(Bvis, batch_size, 1));
        % Now use the reconstructed signal to resample the hidden
        % activations (Gibbs sampling)
        neg_hidden_activations = hidden_function_handle(neg_output_activations * W + repmat(Bhid, batch_size, 1));
        % Get the negative gradient
        neg_gradient = neg_output_activations' * neg_hidden_activations;

        %% Update weights and biases
        W = W + learning_rate * ( (pos_gradient - neg_gradient) / batch_size - regularizer * W );
        
        % Bias update for visible units
        pos_visible_activations = sum(Xb);
        neg_visible_activations = sum(neg_output_activations);
        Bvis = Bvis + learning_rate * (pos_visible_activations - neg_visible_activations) / batch_size;
        
        % Bias update for hidden units
        pos_hiddden_activations = sum(pos_hidden_activations);
        neg_hidden_activations = sum(neg_hidden_activations);
        Bhid = Bhid + learning_rate * (pos_hiddden_activations - neg_hidden_activations) / batch_size;

        %% Compute error
        err = err + sse(Xb - neg_output_activations);
    end
    
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
        plot(h1, 1:epoch, perf(1:epoch), '-*w', 'LineWidth', 1.5)
        xlim(h1, [0.9 epoch+1.1])
        ylim(h1, [0 perf(1)])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance (MSE)')
        set(h1, 'color', [0 0 0])
        
        % Show first image
        wh = sqrt(num_visible);
        imshow(reshape(Xb(1,:)', [wh wh]), 'parent', h2)
        title(h2, 'Image')
        
        % Show reconstruction
        imshow(reshape(neg_output_activations(1,:)', [wh wh]), 'parent', h3)
        title(h3, 'Reconstruction')
        
        % Show first neuron, if possible
        if round(wh) == wh
            imshow(reshape(W(:,1), [wh wh]), 'parent', h4)
            title(h4, 'First unit')
        end
        
        % Update figures
        colormap gray
        drawnow
    end
end

%% Create output
enc = create_layer(num_visible, num_hidden, 'logsig', W', Bhid', 'trainscg', 'Name', 'Encoder');
dec = create_layer(num_hidden, num_visible, 'logsig', W, Bvis', 'trainscg', 'Name', 'Decoder');

end

%% Transfer functions
% Linear
function Y = purelin(X)
Y = X;
end

% Logistic function
function Y = logsig(X)
Y = 1 ./ (1 + exp(-X));
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