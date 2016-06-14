function [enc,dec] = train_ae(X, num_hidden, varargin)
% TRAIN_AE  Train an autoencoder
%   [enc,dec] = TRAIN_AE(X, num_hidden, ...) trains an AE on the data X
%   using a hidden layer of size num_hidden. The result is returned as a
%   pair where enc is the encoder part and dec is the decoder. These can be
%   stacked in an AE using the MATLAB function STACK.
%
%   The data in X is expected to be row-major, meaning that all the feature
%   vectors are stacked as rows on top of each other. If your data is in
%   column-major, use the 'RowMajor' option described below.
%
%   Name value pair options (default value):
%
%       'EncoderFunction' ('logsig'): transfer function applied to the
%       visible units, can be 'logsig', 'tansig' 'purelin', 'poslin' and
%       'satlin'
%
%       'DecoderFunction' ('logsig'): transfer function applied to the
%       hidden units
%
%       'TiedWeights' (true): use tied (symmetric) weights for the
%       encoder/decoder
%
%       'MaxEpochs' (50): number of training iterations
%
%       'Loss' ('mse'): loss function for the output, can be
%       'crossentropy', 'log' (equal to crossentropy), 'mse', 'mae',
%       'binary_crossentropy', 'crossentropy_binary'
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
%       'MaskingNoise' (0): turn the autoencoder into a denoising
%       autoencoder by introducing masking noise (randomly setting inputs
%       to zero) in the interval [0,1[
%
%       'GaussianNoise' (0): turn the autoencoder into a denoising
%       autoencoder by introducing Gaussian noise with a standard deviation
%       as provided
%
%       'DropoutRate' (0): Set this to a number in ]0,1[ for dropout
%       in the hidden layer
%
%       'LearningRate' (0.1): learning rate
%
%       'Momentum' (0.9): momentum
%
%       'Regularizer' (0.0005): regularizer for the weight update
%
%       'Sigma' (0): standard deviation for the random Gaussian
%       distribution used for initializing the weights - if set to zero, it
%       is automatically determined
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
%       'UseGPU' (false): set to true to use GPU if available - note that
%       this demotes all datatypes to single precision floats
%
%       See also TRAIN_RBM.

%% Parse inputs
% Set opts
p = inputParser;
p.CaseSensitive = false;
p.addParameter('EncoderFunction', 'logsig', @ischar)
p.addParameter('DecoderFunction', 'logsig', @ischar)
p.addParameter('TiedWeights', true, @islogical)
p.addParameter('MaxEpochs', 50, @isnumeric)
p.addParameter('Loss', 'mse', @ischar)
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('MaskingNoise', 0, @isnumeric)
p.addParameter('GaussianNoise', 0, @isnumeric)
p.addParameter('DropoutRate', 0, @isnumeric)
p.addParameter('LearningRate', 0.1, @isnumeric)
p.addParameter('Momentum', 0.9, @isnumeric)
p.addParameter('Regularizer', 0.0005, @isnumeric)
p.addParameter('Sigma', 0, @isnumeric)
p.addParameter('RowMajor', true, @islogical)
p.addParameter('Width', 0, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.addParameter('UseGPU', false, @islogical)
p.parse(varargin{:});
% Get opts
encoder_function = p.Results.EncoderFunction;
decoder_function = p.Results.DecoderFunction;

tied_weights = p.Results.TiedWeights;

max_epochs = p.Results.MaxEpochs;

loss = p.Results.Loss;
        
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
momentum = p.Results.Momentum;

row_major = p.Results.RowMajor;
width = p.Results.Width;

verbose = p.Results.Verbose;
visualize = p.Results.Visualize;

use_gpu = p.Results.UseGPU;
% Transpose data to ensure row-major
if ~row_major, X = X'; end
% Get unit function
funs = {'logsig', 'tansig', 'purelin', 'poslin', 'satlin'};
assert(any(strcmpi(encoder_function, funs)) > 0, 'Unknown encoder function: %s!\n', encoder_function);
assert(any(strcmpi(decoder_function, funs)), 'Unknown decoder function: %s!\n', decoder_function);

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
if sigma <= 0
    sigma = 1/sqrt(num_visible);
end
Wvis = sigma * randn(num_hidden, num_visible);
if tied_weights
    Whid = Wvis';
else
    Whid = sigma * randn(num_visible, num_hidden);
end
Bvis = zeros(num_hidden, 1);
Bhid = zeros(num_visible, 1);
Wvisinc = zeros(size(Wvis));
Whidinc = zeros(size(Whid));
Bvisinc = zeros(size(Bvis));
Bhidinc = zeros(size(Bhid));

%% Prepare other stuff
if visualize && max_epochs >= 1
    figname = sprintf('AE %i-%i', num_visible, num_hidden);
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    hfig = figure('Name', figname);
    % If image data
    if width > 0
        h1 = subplot(221);
        h2 = subplot(222);
        h3 = subplot(223);
        h4 = subplot(224);
    else
        h1 = subplot(121);
        h2 = subplot(122);
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
perf = zeros(1,max_epochs);

%% Place data on GPU if possible
has_gpu = (use_gpu && gpuDeviceCount);
if has_gpu
    Xcpu = X;
    X = gpuArray(single(X));
    Wvis = gpuArray(single(Wvis));
    Whid = gpuArray(single(Whid));
    Bvis = gpuArray(single(Bvis));
    Bhid = gpuArray(single(Bhid));
    Wvisinc = gpuArray(single(Wvisinc));
    Whidinc = gpuArray(single(Whidinc));
    Bvisinc = gpuArray(single(Bvisinc));
    Bhidinc = gpuArray(single(Bhidinc));
    perf = gpuArray(single(perf));
    if Nval > 0
        Xval = gpuArray(single(Xval));
        perf_val = gpuArray(single(perf_val));
    end
end

%% Verbosity
if verbose && max_epochs >= 1
    fprintf('****************************************************************************\n');
    fprintf('Training a %i-%i AE using %i training examples\n', num_visible, num_hidden, N);
    if Nval > 0
        fprintf('Using %i/%i batches for training/validation\n', Nbatch, Nval);
    else
        fprintf('Using %i training batches\n', Nbatch);
    end
    fprintf('Using encoder and decoder functions ''%s'' and ''%s''\n', encoder_function, decoder_function);
    fprintf('Using loss function: %s\n', loss);
    if tied_weights, fprintf('Using tied weights\n'), end
    if mask_noise > 0, fprintf('Using a DAE masking noise level of: %g\n', mask_noise); end
    if gauss_noise > 0, fprintf('Using a DAE Gaussian noise level of: %g\n', gauss_noise); end
    if dropout > 0, fprintf('Using a dropout rate of: %g\n', dropout); end
    if has_gpu, fprintf('Using GPU\n'), end
    fprintf('****************************************************************************\n');
end

%% Train
grad = zeros(1,max_epochs);
for epoch = 1:max_epochs
    % Verbosity
    if verbose
        tic
        fprintf('********** Epoch %i/%i (lr: %g, mom: %g, reg: %g, drop: %g) ********** \n', epoch, max_epochs, learning_rate, momentum, regularizer, dropout);
    end
    
    % Shuffle X
    order = randperm(size(X,1));
    X = X(order,:);
    
    % Loop over batches
    train_numel = 0;
    gradbatch = zeros(1, Nbatch);
    for i = 1:Nbatch
        %% Get batch data
        Xb = X(batches{i},:);
        batch_size = size(Xb,1);
        train_numel = train_numel + numel(Xb);
        in = Xb;
        
        %% The DAE case
        if mask_noise > 0
            mask = binornd(1, 1-mask_noise, size(in));
            in = mask .* in;
        end
        
        if gauss_noise > 0
            in = in + gauss_noise * randn(size(in));
        end

        %% Forward pass, hiddens
        ahid = bsxfun(@plus, Wvis*in', Bvis);
        hid = feval(encoder_function, ahid); % Hidden
        dhid = feval(encoder_function, 'dn', ahid); % Hidden derivatives
        
        %% The dropout case
        % https://gist.github.com/ottokart/ebd3d32438c13a62ea3c
        if dropout > 0
            mask = binornd(1, 1-dropout, size(hid));
            hid = mask .* hid / (1-dropout);
            dhid = mask .* dhid / (1-dropout);
        end
        
        %% Forward pass, outputs
        aout = bsxfun(@plus, Whid*hid, Bhid);
        out = feval(decoder_function, aout); % Output
        dout = feval(decoder_function, 'dn', aout); % Output derivatives
        
        %% Backward pass
        [~,derr] = backprop_loss(Xb', out, loss, 'Normalization', 'batch');
        derr = -derr; % Negated output error
        deltaout = dout .* derr; % Output delta
        dhw = deltaout * hid'; % Hidden-output weight gradient
        dhb = sum(deltaout, 2); % Hidden-output bias gradient
        deltahid = dhid .* (Whid' * deltaout); % Hidden delta
        diw = deltahid * in; % Input-hidden weight gradient
        dib = sum(deltahid, 2); % Input-hidden bias gradient
        
%         %% TODO: The CAE case
%         cratio = 0.1;
%         dhidsumsq = sum(dhid.^2, 2); % Column vector of row sums of squared derivatives
%         ddhidsq = 2 * hid.^2 .* (1-hid).^2 .* (1-2*hid);
%         Wvissumsq = sum(Wvis.^2, 2); % Column vector of row sums of squares
%         Wvispen = 2 * repmat(dhidsumsq, 1, num_visible) .* Wvis + repmat(Wvissumsq, 1, num_visible) .* (ddhidsq * in);
%         diw = diw - cratio * Wvispen / batch_size;
        
%         %% TODO: The new penalty case
%         rate = 1e-6;
%         dots = Wvis * Wvis';
%         dots_sign = sign(dots) .* ~eye(num_hidden); % Take sign of dots and remove diagonal
%         Wvispen = dots_sign * Wvis;
%         diw = diw - rate * Wvispen;
        
        %% The tied weight case
        if tied_weights
            diw = diw + dhw';
            dhw = diw';
        end

        %% Update weights and biases
        Wvisinc = momentum * Wvisinc + learning_rate * ( diw - (regularizer / batch_size) * Wvis );
        Wvis = Wvis + Wvisinc;
        
        % Bias update for visible units
        Bvisinc = momentum * Bvisinc + learning_rate * dib;
        Bvis = Bvis + Bvisinc;
        
        Whidinc = momentum * Whidinc + learning_rate * ( dhw - (regularizer / batch_size) * Whid );
        Whid = Whid + Whidinc;
        
        % Bias update for hidden units
        Bhidinc = momentum * Bhidinc + learning_rate * dhb;
        Bhid = Bhid + Bhidinc;
        
        % Compute gradient norm
        gradbatch(i) = gather( norm([diw(:) ; dib(:) ; dhw(:) ; dhb(:)]) );
    end % End loop over batches
    
    % Compute training error (must be done on CPU due to memory limits)
    if has_gpu
        in = Xcpu;
    else
        in = X;
    end
    ahid = bsxfun(@plus, gather(Wvis)*in', gather(Bvis));
    hid = feval(encoder_function, ahid); % Hidden
    aout = bsxfun(@plus, gather(Whid)*hid, gather(Bhid));
    out = feval(decoder_function, aout); % Output
    
    perf(epoch) = backprop_loss(in', out, loss);
    grad(epoch) = mean(gradbatch);
    
    % Compute validation error
    if Nval > 0
        % Pass the validation set through the autoencoder
        ahid = bsxfun(@plus, Wvis*Xval', Bvis);
        hid = feval(encoder_function, ahid); % Hidden
        aout = bsxfun(@plus, Whid*hid, Bhid);
        valout = feval(decoder_function, aout); % Output
        perf_val(epoch) = backprop_loss(Xval', valout, loss);
    end
    
    % Verbosity
    if verbose
        if Nval > 0
            fprintf('Training/validation error: %f/%f\n', perf(epoch), perf_val(epoch));
        else
            fprintf('Training error: %f\n', perf(epoch));
        end
        fprintf('Training time: %.2f s\n', toc);
        fprintf('******************************\n');
    end
    
    % Break cleanly if user closed window
    if ~ishandle(h1), break; end
    
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
        ymax = gather( max(perf) );
        if Nval > 0, ymax = max(ymax, gather(max(perf_val))); end
        ylim(h1, [0 1.1*ymax+eps])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Loss')
            
        % Plot gradient norm of current epoch
        semilogy(h2, 1:epoch, grad(1:epoch), '-k', 'LineWidth', 1.5)
        xlim(h2, [0.5 epoch+0.5])
        if epoch > 1, set(h2, 'xtick', [1 epoch]); end
        ylim(h2, [0.5*min(grad(1:epoch)) inf])
        xlabel(h2, 'Epoch');
        ylabel(h2, 'Gradient norm')
        
        % If image data
        if width > 0
            % Show first image
            imagesc(reshape(in(1,:)', [height width]), 'parent', h3)
            colorbar(h3)
            title(h3, 'Image')
            axis(h3, 'equal', 'off')
            
            % Show reconstruction
            imagesc(reshape(out(:,1), [height width]), 'parent', h4)
            colorbar(h4)
            title(h4, 'Reconstruction')
            axis(h4, 'equal', 'off')
            
            colormap gray
            
            % Show the strongest/weakest neurons
            plot_neurons(gather(Wvis'), width, 5, 'Strongest', true);
            plot_neurons(gather(Wvis'), width, 5, 'Strongest', false);
        end
        
        % Update figures
        drawnow
    end % End visualization
end % End loop over epochs

if exist('hfig', 'var'), print(hfig, figname, '-dpdf'); end

%% Create output
enc = create_layer(num_visible, num_hidden, encoder_function, double(gather(Wvis)), double(gather(Bvis)), 'trainscg', 'Name', 'Encoder');
dec = create_layer(num_hidden, num_visible, decoder_function, double(gather(Whid)), double(gather(Bhid)), 'trainscg', 'Name', 'Decoder');

%% Clean up
if has_gpu
    clear X Wvis Whid Bvis Bhid Wvisinc Whidinc Bvisinc Bhidinc perf;
    if Nval > 0
        clear Xval perf_val;
    end
end

end
