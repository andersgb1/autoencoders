function [enc,dec] = train_rbm(X, num_hidden, varargin)
% TRAIN_RBM  Train a restricted Boltzmann machine
%   rbm = TRAIN_RBM(X, num_hidden, ...) trains an RBM on the data X using a
%   hidden layer of size num_hidden.
%
%   Name value pair options (default value):
%
%       'RowMajor' (true): logical specifying whether the observations in X
%       are placed in rows or columns
%
%       'MaxEpochs' (50): number of training iterations
%
%       'LearningRate' (0.1): learning rate
%
%       'Verbose' (false): logical, set to true to print status messages
%
%       'Visualize' (false): logical, set to true to show status plots
%

%% Parse inputs
p = inputParser;
p.CaseSensitive = false;
% Set opts
p.addOptional('RowMajor', true, @islogical)
p.addOptional('MaxEpochs', 50, @isnumeric)
p.addOptional('NumBatches', 100, @isnumeric)
p.addOptional('LearningRate', 0.1, @isfloat)
p.addOptional('Verbose', false, @islogical)
p.addOptional('Visualize', false, @islogical)
p.parse(varargin{:});
% Get opts
row_major = p.Results.RowMajor;
max_epochs = p.Results.MaxEpochs;
num_batches = p.Results.NumBatches;
learning_rate = p.Results.LearningRate;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;

% Transpose data to ensure row-major
if ~row_major, X = X'; end

%% Initialize weights and biases
[N, num_visible] = size(X);
W = 0.1 * randn(num_visible, num_hidden);
Bvis = zeros(1, num_visible);
Bhid = zeros(1, num_hidden);
% % Insert biases
% W = [zeros(1,num_hidden) ; W];
% W = [zeros(num_visible+1,1) W];
% % Insert bias of 1 in front of each observation
% X = [ones(N,1) X];

%% Prepare other stuff
if visualize
    figname = 'RBM';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    figure('Name', figname)
    h1 = subplot(131);
    h2 = subplot(132);
    h3 = subplot(133);
end

%% Setup mini-batches
Nbatch = 1;
if num_batches > 1, Nbatch = floor(N / num_batches); end

%% Train
perf = zeros(1,max_epochs);
for epoch = 1:max_epochs
    % Verbosity
    if verbose
        tic
        fprintf('********** Epoch %i/%i **********\n', epoch, max_epochs);
    end
    
    % Loop over batches
    error = 0;
    chars = 0;
    for batch_begin = 1:Nbatch:N
        batch_end = min([batch_begin + Nbatch - 1, N]);
        batch_size = batch_end - batch_begin + 1;
        
        % Verbosity
        if verbose
            for i = 1:chars, fprintf('\b'); end
            chars = fprintf('%i/%i\n', batch_end, N);
        end
        
        % Get batch data
        if batch_end > N
            Xb = X(batch_begin:end,:);
        else
            Xb = X(batch_begin:batch_end,:);
        end
        
        % Clamp to the data and sample from the hidden units. 
        % (This is the "positive CD phase", aka the reality phase.)
        pos_hidden_activations = Xb * W;
    %     pos_hidden_probs = logistic(pos_hidden_activations);
        pos_hidden_probs = logistic(pos_hidden_activations + repmat(Bhid, batch_size, 1));
        pos_hidden_states = pos_hidden_probs > rand(batch_size, num_hidden);
        % Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
        % themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
        % "A Practical Guide to Training Restricted Boltzmann Machines" for more.
        pos_associations = Xb' * pos_hidden_probs;

        % Reconstruct the visible units and sample again from the hidden units.
        % (This is the "negative CD phase", aka the daydreaming phase.)
        neg_visible_activations = pos_hidden_states * W';
        neg_visible_probs = logistic(neg_visible_activations + repmat(Bvis, batch_size, 1));
    %     neg_visible_probs(:,1) = 1; % Fix the bias unit
        neg_hidden_activations = neg_visible_probs * W;
    %     neg_hidden_probs = logistic(neg_hidden_activations);
        neg_hidden_probs = logistic(neg_hidden_activations + repmat(Bhid, batch_size, 1));
        % Note, again, that we're using the activation *probabilities* when computing associations, not the states
        % themselves.
        neg_associations = neg_visible_probs' * neg_hidden_probs;

        % Update weights
        W = W + learning_rate * (pos_associations - neg_associations) / batch_size;

        pos_vis_act = sum(Xb);
        neg_vis_act = sum(neg_visible_probs);
        Bvis = Bvis + learning_rate * (pos_vis_act - neg_vis_act) / batch_size;

        pos_hid_act = sum(pos_hidden_probs);
        neg_hid_act = sum(neg_hidden_probs);
        Bhid = Bhid + learning_rate * (pos_hid_act - neg_hid_act) / batch_size;

        % Compute error
        error = error + sse(Xb - neg_visible_probs);
    end
    
    % Store performance
    perf(epoch) = error/numel(X);
    
    % Verbosity
    if verbose
        fprintf('Error (SSE/MSE): %.0f/%.5f\n', error, perf(epoch));
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
        imshow(reshape(Xb(1,:)', [wh wh]), 'parent', h2) % Avoid the bias unit
        title(h2, 'Image')
        
        % Show reconstruction
        imshow(reshape(neg_visible_probs(1,:)', [wh wh]), 'parent', h3) % Avoid the bias unit
        title(h3, 'Reconstruction')
        
        % Update figures
        colormap gray
        drawnow
    end
end

%% Create output
enc = create_layer(num_visible, num_hidden, 'logsig', W', Bhid', 'trainscg', 'Name', 'Encoder');
dec = create_layer(num_hidden, num_visible, 'logsig', W, Bvis', 'trainscg', 'Name', 'Decoder');

end

%% Logistic function
function Y = logistic(X)
Y = 1 ./ (1 + exp(-X));
end
