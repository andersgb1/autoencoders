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
%       'VisibleFunction' ('purelin'): transfer function for the visible
%       units in the first layer, can be 'logsig', 'tansig', 'purelin',
%       etc.
%
%       'HiddenFunction' ('logsig'): transfer function for the hidden
%       units in all layers
%
%       'OutputFunction' ('purelin'): transfer function for the hidden
%       units in the final layer
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
%       'BatchesInit' (empty cell): mini-batches considered in each epoch
%       of pretraining. If you want to split the training data into
%       mini-batches during each epoch, this argument should contain a cell
%       array, each element being indices for a mini-batch.
%
%       'Batches' (empty cell): mini-batches considered in each epoch
%       of backpropagation.
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
%       'TrainFcn' ('trainscg'): training function to use during
%       backpropagation
%
%       'RowMajor' (true): logical specifying whether the observations in X
%       are placed in rows or columns
%
%       'Verbose' (false): logical, set to true to print status messages
%
%       'Visualize' (false): logical, set to true to show status plots
%
%       'Resume' (false): logical, if set to true, allow for resuming the
%       pretraining. This means that during the
%
%       See also TRAIN_RBM.

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('VisibleFunction', 'purelin', @ischar)
p.addParameter('HiddenFunction', 'logsig', @ischar)
p.addParameter('OutputFunction', 'purelin', @ischar)
p.addParameter('UnitFunction', 'default', @ischar)
p.addParameter('MaxEpochsInit', 50, @isnumeric)
p.addParameter('MaxEpochs', 200, @isnumeric)
p.addParameter('BatchesInit', {}, @iscell)
p.addParameter('Batches', {}, @iscell)
p.addParameter('LearningRate', 0.1, @isfloat)
p.addParameter('LearningRateMul', 1, @isfloat)
p.addParameter('Momentum', 0.5, @isfloat)
p.addParameter('MomentumInc', 0, @isfloat)
p.addParameter('Regularizer', 0.0005, @isfloat)
p.addParameter('Sigma', 0.1, @isfloat)
p.addParameter('TrainFcn', 'trainscg', @ischar)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.addParameter('Resume', false, @islogical)
p.parse(varargin{:});
% Get opts
visible_function = p.Results.VisibleFunction;
hidden_function = p.Results.HiddenFunction;
output_function = p.Results.OutputFunction;
unit_function = p.Results.UnitFunction;
max_epochs_init = p.Results.MaxEpochsInit;
max_epochs = p.Results.MaxEpochs;
batches_init = p.Results.BatchesInit;
batches = p.Results.Batches;
regularizer = p.Results.Regularizer;
sigma = p.Results.Sigma;
train_fcn = p.Results.TrainFcn;
learning_rate = p.Results.LearningRate;
learning_rate_mul = p.Results.LearningRateMul;
momentum = p.Results.Momentum;
momentum_inc = p.Results.MomentumInc;
row_major = p.Results.RowMajor;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
resume = p.Results.Resume;
% Transpose data to ensure row-major
if ~row_major, X = X'; end

%% Start pretraining
if resume && exist('net', 'var')
    if verbose, fprintf('Pretrained network already exists! Skipping pretraining...\n'); end
else
    inputs = X;
    enc_init = [];
    dec_init = [];
    for i = 1:length(num_hidden)
        numhid = num_hidden(i);
        hidfun = hidden_function;
        learnrate = learning_rate;
        if i == 1
            visfun = visible_function;
        elseif i == length(num_hidden)
            visfun = hidden_function;
            hidfun = output_function;
        else
            visfun = hidden_function;
            hidfun = hidden_function;
        end
        
        funs = {'purelin', 'poslin', 'satlin', 'satlins'};
        if any( [strcmpi(visfun, funs) strcmpi(hidfun,  funs)] )
            learnrate = learning_rate / 100;
            warning('Linear visible/hidden RBM units selected! Scaling learning rate: %.6f --> %.6f...\n', learning_rate, learnrate);
        end
        
        rbmfile = sprintf('rbm%i.mat', i);
        if resume && exist(rbmfile, 'file')
            if verbose, fprintf('Loading RBM %i from file...\n', i); end
            load(rbmfile);
        else
            [enci,deci] = train_rbm(inputs, numhid,...
                'VisibleFunction', visfun,...
                'HiddenFunction', hidfun,...
                'UnitFunction', unit_function,...
                'MaxEpochs', max_epochs_init,...
                'Batches', batches_init,...
                'LearningRate', learnrate,...
                'Momentum', momentum,...
                'Regularizer', regularizer,...
                'Sigma', sigma,...
                'Verbose', verbose,...
                'Visualize', visualize);
            if resume, save(rbmfile, 'enci', 'deci'); end
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
    
    %% Stack the RBMs
    net_init = stack(enc_init, dec_init);
    %     net_init.divideFcn = 'dividetrain';
    net_init.performFcn = 'mse';
    %     net_init.performParam.regularization = 0;
    %     net_init.performParam.normalization = 'none';
    %     net_init.plotFcns = {'plotperform'};
    %     net_init.plotParams = {nnetParam}; % Dummy?
        net_init.trainFcn = train_fcn;
        net_init.trainParam.epochs = max_epochs;
    %     net_init.trainParam.showWindow = visualize;
    %     net_init.trainParam.showCommandLine = verbose;
    %     net_init.trainParam.show = 1;
    net = net_init;
end

%% Start fine tuning
if verbose
    fprintf('****************************************************************************\n');
    fprintf('Fine tuning the DBN for %i epochs using training function ''%s''\n', max_epochs, train_fcn);
    fprintf('****************************************************************************\n');
end



% net = train(net, X', X');


%% Setup mini-batches (mulbatch times larger than the RBM mini-batches)
N = size(X,1);
if isempty(batches)
    batches = {1:N};
end

%% Resume
iter_start = 1;
netfile = 'net.mat';
if resume && exist(netfile, 'file')
    load(netfile);
    iter_start = iter;
    
    % Update learning rate and momentum
    for i = 1:(iter-1)
        learning_rate = learning_rate * learning_rate_mul;
        momentum = momentum + momentum_inc;
    end
    
    if iter_start < max_epochs && verbose, fprintf('Resuming fine tuning from epoch %i...\n', iter_start); end
end

%% Prepare other stuff
if iter_start < max_epochs && visualize
    figname = 'DBN';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    hfig = figure('Name', figname);
    % If image data
    wh = sqrt(size(X,2));
    if wh == round(wh)
        h1 = subplot(231);
        h2 = subplot(232);
        h3 = subplot(233);
        h4 = subplot(234);
        h5 = subplot(235);
        h6 = subplot(236);
    else
        h1 = subplot(221);
        h2 = subplot(222);
        h4 = subplot(223);
        h5 = subplot(224);
    end
end

%% Start backpropagation
perf = zeros(1, max_epochs);
grad = zeros(1, max_epochs);
Winc = 0;
for epoch = iter_start:max_epochs
    % Verbosity
    if verbose, fprintf('Epoch %i/%i (lr: %f, mom: %f)\n', epoch, max_epochs, learning_rate, momentum); end
    Nbatch = length(batches);
    perfbatch = zeros(1, Nbatch);
    gradbatch = zeros(1, Nbatch);
    for j=1:Nbatch
        % Verbosity
        if verbose, chars = fprintf('\tBatch %i/%i\n', j, Nbatch); end
        
        % Get batch data
        Xb = X(batches{j},:);
        
        % Get current weights
        w = getwb(net);
        
        % Run minimization
%         options = zeros(1,18);
%         options(1) = -1; % Display SCG progress?
%         options(14) = 3; % Maximum SCG iterations
%         [w, ~, flog] = scg(@f, w', options, @df, net, Xb');

        gradj = df(w, net, Xb')';
        Winc = momentum * Winc - learning_rate * gradj - learning_rate * regularizer * w;
        w = w + Winc;
        
%         % Compute error of mini-batch
%         perfbatch(j) = flog(end); % MSE for current batch
        perfbatch(j) = f(w, net, Xb');
        gradbatch(j) = norm(gradj);
        
        % Update weights
        net = setwb(net, w);
        
        % Verbosity
        if verbose
            chars = chars + fprintf('\tReconstruction error of mini-batch (MSE): %f\n', perfbatch(j));
        end
        
        % Visualization
        if visualize
            % h1 is shown below
            
            % Plot performance of current mini-batch
            plot(h2, 1:j, perfbatch(1:j), '-k', 'LineWidth', 1.5)
            xlim(h2, [0.9 j+1.1])
            ylim(h2, [0 inf])
            xlabel(h2, 'Mini-batch');
            ylabel(h2, 'Performance (MSE)')
            
            % Plot gradient norm of current mini-batch
            plot(h5, 1:j, gradbatch(1:j), '-k', 'LineWidth', 1.5)
            xlim(h5, [0.9 j+1.1])
            ylim(h5, [0 inf])
            xlabel(h5, 'Mini-batch');
            ylabel(h5, 'Gradient (L2 norm)')
            
            % If image data
            if round(wh) == wh
%                 % Show first neuron
%                 W = net.IW{1}';
%                 imagesc(reshape(W(:,1), [wh wh]), 'parent', h4)
%                 colorbar(h4)
%                 title(h4, 'First unit')
%                 axis equal
%                 axis off
                
                % Show first image
                imshow(reshape(X(1,:)', [wh wh]), 'parent', h3)
                title(h3, 'Image')
                
                % Show reconstruction
                imshow(reshape(net(X(1,:)'), [wh wh]), 'parent', h6)
                title(h6, 'Reconstruction')
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
    perf(epoch) = perform(net, X', net(X')); % MSE
    
    % Visualization
    if visualize
        % Plot performance of current epoch
        plot(h1, iter_start:epoch, perf(iter_start:epoch), '-*k', 'LineWidth', 1.5)
        xlim(h1, [iter_start-0.9 epoch+1.1])
        ylim(h1, [0 inf])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance (MSE)')
        
        w = getwb(net);
        grad(epoch) = norm( df(w, net, X') );
        plot(h4, iter_start:epoch, grad(iter_start:epoch), '-*k', 'LineWidth', 1.5)
        xlim(h4, [iter_start-0.9 epoch+1.1])
        ylim(h4, [0 inf])
        xlabel(h4, 'Epoch')
        ylabel(h4, 'Gradient (L2 norm)')
            
        % Update figures
        colormap(hfig, 'gray');
        drawnow
    end
    
    % Save state
    if resume
        if verbose, fprintf('Saving fine tuned net for epoch %i...\n', epoch); end
        iter = epoch + 1; % Save instead the next epoch index
        save(netfile, 'net', 'iter');
    end
end % End epochs



%% Set outputs
% Get encoder
if nargout > 1
    enc = [];
    for epoch = 1:length(num_hidden)
        if epoch == 1, enc = get_layer(net, 1); else enc = stack(enc, get_layer(net, epoch)); end
    end
    varargout{1} = enc;
end

% Get decoder
if nargout > 2
    dec = [];
    for epoch = (length(num_hidden)+1):(2*length(num_hidden))
        if epoch == (length(num_hidden)+1), dec = get_layer(net, (length(num_hidden)+1)); else dec = stack(dec, get_layer(net, epoch)); end
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

function err = f(w, net, X)
wnet = setwb(net, w);
err = perform(wnet, X, wnet(X));
end

function grad = df(w, net, X)
wnet = setwb(net, w);
grad = -defaultderiv('dperf_dwb', wnet, X, X)';
end
