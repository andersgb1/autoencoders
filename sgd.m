function net = sgd(net, input, target, max_epochs, learning_rate, varargin)

%% Get opts
if nargin < 4, max_epochs = 100; end
if nargin < 5, learning_rate = 1; end
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('Loss', 'crossentropy', @ischar)
p.addParameter('Optimizer', 'sgd', @ischar)
p.addParameter('AnnealRate', 1, @isnumeric)
p.addParameter('Momentum', 0, @isnumeric)
p.addParameter('Regularizer', 0, @isnumeric)
p.addParameter('DropoutRate', 0, @isnumeric)
p.addParameter('MinGrad', 1e-10, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('TestData', struct(), @isstruct)
p.addParameter('Visualize', false, @islogical)
p.addParameter('CheckpointFile', 'sgd.mat', @ischar)
p.addParameter('UseGPU', false, @islogical)
p.parse(varargin{:});
batches = p.Results.Batches;
val_frac = p.Results.ValidationFraction;
assert(val_frac >= 0 && val_frac < 1, 'Validation fraction must be a number in [0,1[!')
loss = p.Results.Loss;
optimizer = p.Results.Optimizer;
% TODO: Correct this syntax
adadelta = false;
adam = false;
if strcmpi(optimizer, 'sgd')
    ;
elseif strcmpi(optimizer, 'adadelta')
    adadelta = true;
elseif strcmpi(optimizer, 'adam')
    adam = true;
else
    error('Unknown optimizer: %s!', optimizer);
end
anneal = p.Results.AnnealRate;
assert(anneal > 0, 'Learning rate annealing factor must be positive!')
momentum = p.Results.Momentum;
dropout = p.Results.DropoutRate;
assert(dropout < 1, 'Dropout rate must be < 1!');
regularizer = p.Results.Regularizer;
min_grad = p.Results.MinGrad;
verbose = p.Results.Verbose;
test_data = p.Results.TestData;
if ~isempty(test_data),
    assert(isfield(test_data, 'input') && isfield(test_data, 'target'),...
        'Test data must be a struct with fields input and target!');
end
visualize = p.Results.Visualize;
checkpoint_file = p.Results.CheckpointFile;
use_gpu = p.Results.UseGPU;

% TODO: Standardize the dataset depending on the units of the first layer
input_function = net.layers{1}.transferFcn;
if strcmpi(input_function, 'purelin')
    warning('Linear input units selected! Mean subtracting the dataset...');
    meanin = mean(input, 2);
    input = bsxfun(@minus, input, meanin);
    if ~isempty(test_data)
        test_data.input = bsxfun(@minus, test_data.input, meanin);
    end
elseif any(strcmpi(input_function, {'logsig', 'satlin'}))
    warning('Logistic sigmoid or saturated linear input units selected! Normalizing dataset to [0,1]...');
    input = (input - min(input(:))) / (max(input(:)) - min(input(:)));
    if ~isempty(test_data)
        test_data.input = (test_data.input - min(test_data.input(:))) / (max(test_data.input(:)) - min(test_data.input(:)));
    end
elseif any(strcmpi(input_function, {'tansig', 'satlins'}))
    warning('Tangent sigmoid or symmetric saturated linear input units selected! Normalizing dataset to [-1,1]...');
    input = 2 * (input - min(input(:))) / (max(input(:)) - min(input(:))) - 1;
    if ~isempty(test_data)
        test_data.input = 2 * (test_data.input - min(test_data.input(:))) / (max(test_data.input(:)) - min(test_data.input(:))) - 1;
    end
end

if gpuDeviceCount && use_gpu
    input=gpuArray(input);
    target=gpuArray(target);
end

%% Setup mini-batches
N = size(input,2);
if isempty(batches), batches = {1:N}; end

Nbatch = length(batches);
Nval = 0;
if val_frac > 0
    Nval = round(val_frac * Nbatch);
    if Nval > 0
        Nbatch = Nbatch - Nval;
        batches_val = [batches{(Nbatch+1):(Nbatch+Nval)}]; % Produces a batch_size x Nval matrix
        batches = batches(1:Nbatch); % Produces a cell array
        val_input = input(:,batches_val(:));
        val_target = target(:,batches_val(:));
        perf_val = zeros(1, max_epochs);
    end
end

%% Prepare viz
if visualize
    figname = 'SGD';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    hfig = figure('Name', figname);
    if isempty(test_data), row_plots = 1; col_plots = 3; else row_plots = 2; col_plots = 2; end
    h1 = subplot(row_plots,col_plots,1);
    h2 = subplot(row_plots,col_plots,2);
    h3 = subplot(row_plots,col_plots,3);
    if ~isempty(test_data)
        h4 = subplot(row_plots,col_plots,4);
        % Set a flag high if we are doing classification
        is_classifier = strcmp(net.layers{end}.transferFcn, 'softmax');
    end
end

%% Resume
if exist(checkpoint_file, 'file')
    if verbose, fprintf('Resuming backpropagation using checkpoint %s...\n', checkpoint_file); end
    load(checkpoint_file);
    net = setwb(net, w);
    [W,b,transfer] = get_ffnet_info(net);
else
    iter = 1;
    perf = zeros(1, max_epochs);
    grad = zeros(1, max_epochs);
    perf_incs = 0;
    perf_val_incs = 0;
    test_error = ones(1, max_epochs);
    [W,b,transfer] = get_ffnet_info(net);
    Winc = repmat({0}, 1, length(W));
    binc = repmat({0}, 1, length(b));
    % TODO
%     Wbest = repmat({0}, 1, length(W));
%     bbest = repmat({0}, 1, length(b));
%     perf_best = inf;
    if adadelta
        EWgrad = repmat({0}, 1, length(W));
        EWdelta = repmat({0}, 1, length(W));
        Ebgrad = repmat({0}, 1, length(b));
        Ebdelta = repmat({0}, 1, length(b));
    elseif adam
        adam_mW = repmat({0}, 1, length(W));
        adam_vW = repmat({0}, 1, length(W));
        adam_mb = repmat({0}, 1, length(b));
        adam_vb = repmat({0}, 1, length(b));
    end
end

%% TODO: Get current weights
if gpuDeviceCount && use_gpu
    for k=1:length(W),
        W{k}=gpuArray(W{k});
        b{k}=gpuArray(b{k});
    end
end

%% Start backpropagation
for epoch = iter:max_epochs
    % Anneal the learning rate
    lr = anneal^epoch * learning_rate;
    
    % Verbosity
    if verbose
        tstart = tic;
        if adadelta || adam
            fprintf('********** Epoch %i/%i (mom: %g, reg: %g, drop: %g) ********** \n', epoch, max_epochs, momentum, regularizer, dropout);
        else
            fprintf('********** Epoch %i/%i (lr: %g, mom: %g, reg: %g, drop: %g) ********** \n', epoch, max_epochs, lr, momentum, regularizer, dropout);
        end
    end
    
    
    % Shuffle inputs
    order = randperm(Nbatch);
    batches = batches(order);
    
    % Loop over batches
    lossbatch = zeros(1, Nbatch);
    gradbatch = zeros(1, Nbatch);
    
    for j=1:Nbatch
        % Get batch data
        batch_input = input(:, batches{j});
        batch_target = target(:, batches{j});
        batch_size = size(target, 2);
        
        % Run minimization
        [dW,db,lossj] = backprop(W, b, transfer, batch_input, batch_target, loss,...
            'DropoutRate', dropout, 'Normalization', 'batch');
        for k=1:length(W)
            if adadelta
                % Adadelta step, weights
                EWgrad{k} = momentum * EWgrad{k} + (1 - momentum) * dW{k}.^2;
                Winc{k} = -sqrt(EWdelta{k} + 1e-6) ./ sqrt(EWgrad{k} + 1e-6) .* dW{k};
                EWdelta{k} = momentum * EWdelta{k} + (1 - momentum) * Winc{k}.^2;
                % Adadelta step, biases
                Ebgrad{k} = momentum * Ebgrad{k} + (1 - momentum) * db{k}.^2;
                binc{k} = -sqrt(Ebdelta{k} + 1e-6) ./ sqrt(Ebgrad{k} + 1e-6) .* db{k};
                Ebdelta{k} = momentum * Ebdelta{k} + (1 - momentum) * binc{k}.^2;
            elseif adam
                alpha = 0.001;
                beta1 = momentum;
                beta2 = 0.999;
                t = (epoch - 1) * j + 1; % TODO: Use global or local timestep?
%                 t = j; % Local timestep
                % Adam step, weights
                adam_mW{k} = beta1 * adam_mW{k} + (1 - beta1) * dW{k};
                adam_vW{k} = beta2 * adam_vW{k} + (1 - beta2) * dW{k}.^2;
                mhat = adam_mW{k} ./ (1 - beta1.^t);
                vhat = adam_vW{k} ./ (1 - beta2.^t);
                Winc{k} = -alpha * mhat ./ (sqrt(vhat) + 1e-8);
                % Adam step, biases
                adam_mb{k} = beta1 * adam_mb{k} + (1 - beta1) * db{k};
                adam_vb{k} = beta2 * adam_vb{k} + (1 - beta2) * db{k}.^2;
                mhat = adam_mb{k} ./ (1 - beta1.^t);
                vhat = adam_vb{k} ./ (1 - beta2.^t);
                binc{k} = -alpha * mhat ./ (sqrt(vhat) + 1e-8);
            else
                % SGD step
                Winc{k} = momentum * Winc{k} - lr * dW{k};
                binc{k} = momentum * binc{k} - lr * db{k};
            end
            W{k} = W{k} + Winc{k} - (regularizer / batch_size) * W{k};
            b{k} = b{k} + binc{k};
        end
            
        % Compute error and gradient norm of mini-batch
        lossbatch(j) = gather( lossj);
%         if gpuDeviceCount && use_gpu
            ssewb = 0;
            for k = 1:length(dW)
                ssewb = ssewb + sse(dW{k});
                ssewb = ssewb + sse(db{k});
            end
            gradbatch(j) = gather( sqrt(ssewb) );
%         else
%             gradbatch(j) = sqrt(sum(cellfun(@sse, dW)) + sum(cellfun(@sse, db)));
%         end
        
        % Visualization
        if visualize && ishandle(h1)
            % Plot performance of current mini-batch
            plot(h1, 1:j, lossbatch(1:j), '-k')
            xlim(h1, [0.5 Nbatch+0.5]);
            set(h1, 'xtick', [1 Nbatch]);
            ylim(h1, [0 inf])
            xlabel(h1, 'Mini-batch');
            ylabel(h1, 'Loss')
            
            drawnow
        end
    end % End loop over batches
    
    % Update weights in the network object
    count=1;
    w=zeros(sum(cellfun(@numel, W)) + sum(cellfun(@numel, b)), 1);
    for k=1:length(W)
        bnum = numel(b{k});
        w(count:count+bnum-1) = gather( b{k} );
        count = count + bnum;
        Wnum = numel(W{k});
        w(count:count+Wnum-1) = gather( reshape(W{k}, Wnum, 1) );
        count = count + Wnum;
    end

    net = setwb(net, w);
    
%     % TODO: Update best result
%     if Nval > 0
%         perf_epoch = perf_val(epoch);
%     else
%         perf_epoch = perf(epoch);
%     end
%     
%     if perf_epoch < perf_best
%         Wbest = W;
%         bbest = b;
%     end
    
    % Store performance parameters
    perf(epoch) = gather( backprop_loss(target, prop(W,b,transfer,input), loss) );
    grad(epoch) = mean(gradbatch);
    if ~isempty(test_data)
        test_error(epoch) = confusion(test_data.target, net(test_data.input));
    end
    if Nval > 0, perf_val(epoch) = gather( backprop_loss(val_target, prop(W,b,transfer,val_input), loss) ); end
    
    % Verbosity
    if verbose
        if Nval > 0
            fprintf('Training/validation error: %f/%f\n', perf(epoch), perf_val(epoch));
        else
            fprintf('Training error: %f\n', perf(epoch));
        end
        if ~isempty(test_data)
            fprintf('Test accuracy: %g %%\n', 100*(1-test_error(epoch)));
        end
        fprintf('Training time: %.2f s\n', toc(tstart));
        fprintf('******************************\n');
    end
    
    % Break cleanly if user closed window
    if ~ishandle(h1), break; end
    
    % Visualization
    if visualize
        % Plot performance of current epoch
        semilogy(h2, 1:epoch, perf(1:epoch), '-*k', 'LineWidth', 1.5)
        if Nval > 0
            hold(h2, 'on')
            semilogy(h2, 1:epoch, perf_val(1:epoch), '-sr', 'LineWidth', 1.5)
            legend(h2, 'Training', 'Validation', 'Location', 'best')
            hold(h2, 'off')
        end
        xlim(h2, [0.5 epoch+0.5])
        if epoch > 1, set(h2, 'xtick', [1 epoch]); end
        if Nval > 0
            ylim(h2, [0.5*min([perf(1:epoch) perf_val(1:epoch)]) inf])
        else
            ylim(h2, [0.5*min(perf(1:epoch)) inf])
        end
        xlabel(h2, 'Epoch')
        ylabel(h2, 'Loss')
        if Nval > 0
            title(h2, sprintf('Best val perf: %.2e', min(perf_val(1:epoch))))
        else
            title(h2, sprintf('Best perf: %.2e', min(perf(1:epoch))))
        end
            
        % Plot gradient norm of current epoch
        semilogy(h3, 1:epoch, grad(1:epoch), '-k')
        xlim(h3, [0.5 epoch+0.5])
        if epoch > 1, set(h3, 'xtick', [1 epoch]); end
        ylim(h3, [0.5*min(grad(1:epoch)) inf])
        xlabel(h3, 'Epoch');
        ylabel(h3, 'Gradient norm')
        
        % Plot test error, if available
        if ~isempty(test_data)
            if is_classifier
                test_cases = size(test_data.target, 2);
                test_errors = test_error(1:epoch) * test_cases;
                plot(h4, 1:epoch, test_errors, 'k', 'LineWidth', 1.5)
                xlim(h4, [0.5 epoch+0.5])
                if epoch > 1, set(h4, 'xtick', [1 epoch]); end
                ylim(h4, [0 max(test_errors)+0.5])
                xlabel(h4, 'Epoch')
                ylabel(h4, 'Test errors')
                title(h4, sprintf('Best: %i', round(min(test_errors))))
            else
                test_accuracy_pct = 100-100*test_error(1:epoch);
                plot(h4, 1:epoch, test_accuracy_pct, 'k', 'LineWidth', 1.5)
                xlim(h4, [0.5 epoch+0.5])
                if epoch > 1, set(h4, 'xtick', [1 epoch]); end
                ylim(h4, [min(test_accuracy_pct)-0.5 100])
                xlabel(h4, 'Epoch')
                ylabel(h4, 'Test accuracy [%]')
                title(h4, sprintf('Best: %g %%', max(test_accuracy_pct)))
            end
        end
        
        % Update figures
        drawnow
    end
    
    % Save state
    if ~isempty(checkpoint_file)
        iter = epoch+1;
        save(checkpoint_file, 'iter', 'w', 'Winc', 'binc', 'grad', 'perf', 'perf_incs');
        if Nval > 0
            save(checkpoint_file, 'perf_val', 'perf_val_incs', '-append');
        end
        if ~isempty(test_data)
            save(checkpoint_file, 'test_error', '-append');
        end
        if adadelta
            save(checkpoint_file, 'EWgrad', 'EWdelta', 'Ebgrad', 'Ebdelta', '-append');
        elseif adam
            save(checkpoint_file, 'adam_mW', 'adam_vW', 'adam_mb', 'adam_vb', '-append');
        end
    end
    
    % Termination
    if grad(epoch) < min_grad
        fprintf('Gradient has reached a minimum at %f!', grad(epoch));
        fprintf('\tStopping backpropagation...\n');
        break
    end
    
    % Convergence of training/validation set
    if epoch > 1
        % First update how many times training error has increased
        if perf(epoch) < perf(epoch-1)
            perf_incs = 0;
        else
            perf_incs = perf_incs + 1;
        end
        % Then stop if it happened many times in a row
        if perf_val == 5
            fprintf('Training error has saturated!');
            fprintf('\tStopping backpropagation...\n');
            break;
        end
        % Same thing for validation set
        if Nval > 0
            if perf_val(epoch) < perf_val(epoch-1)
                perf_val_incs = 0;
            else
                perf_val_incs = perf_val_incs + 1;
            end
            % Then stop if it happened many times in a row
            if perf_val_incs == 5
                fprintf('Validation error has saturated!');
                fprintf('\tStopping backpropagation...\n');
                break;
            end
        end
    end
end % End epochs

if exist('hfig', 'var'), print(hfig, figname, '-dpdf'); end
