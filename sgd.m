function net = sgd(net, input, target, max_epochs, learning_rate, varargin)

%% Get opts
if nargin < 4, max_epochs = 100; end
if nargin < 5, learning_rate = 1; end
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('Loss', 'crossentropy', @ischar)
p.addParameter('Adadelta', false, @islogical)
p.addParameter('Momentum', 0, @isnumeric)
p.addParameter('Regularizer', 0, @isnumeric)
p.addParameter('MinGrad', 1e-10, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.addParameter('CheckpointFile', 'sgd.mat', @ischar)
p.parse(varargin{:});
batches = p.Results.Batches;
val_frac = p.Results.ValidationFraction;
assert(val_frac >= 0 && val_frac < 1, 'Validation fraction must be a number in [0,1[!')
loss = p.Results.Loss;
adadelta = p.Results.Adadelta;
momentum = p.Results.Momentum;
regularizer = p.Results.Regularizer;
min_grad = p.Results.MinGrad;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;
checkpoint_file = p.Results.CheckpointFile;

%% Setup mini-batches
N = size(input,2);
if isempty(batches), batches = {1:N}; end

Nbatch = length(batches);
Nval = 0;
if val_frac > 0
    Nval = round(val_frac * Nbatch);
    if Nval > 0
        Nbatch = Nbatch - Nval;
        batches_val = batches{(Nbatch+1):(Nbatch+Nval)}; % Produces a vector
        batches = batches(1:Nbatch); % Produces a cell array
        val_input = input(:,batches_val);
        val_target = target(:,batches_val);
        perf_val = zeros(1, max_epochs);
    end
end

%% Prepare viz
if visualize
    figname = 'SGD';
    if ~isempty(findobj('type', 'figure', 'name', figname)), close(figname); end
    figure('Name', figname);
    h1 = subplot(1,3,1);
    h2 = subplot(1,3,2);
    h3 = subplot(1,3,3);
end

%% Resume
if exist(checkpoint_file, 'file')
    if verbose, fprintf('Resuming backpropagation using checkpoint %s...\n', checkpoint_file); end
    load(checkpoint_file);
    net = setwb(net, w);
else
    iter = 1;
    perf = zeros(1, max_epochs);
    grad = zeros(1, max_epochs);
    Winc = 0;
end

%% Start backpropagation
% lr_dec = 0; % Number of times we decreased the learning rates
% hwait = waitbar(iter/max_epochs, '');
if visualize
    hwait = waitbar(0, '',...
        'Name','Running backpropagation...',...
        'CreateCancelBtn', 'setappdata(gcbf,''canceling'',1)');
end
Egrad = 0;
Edelta = 0;
for epoch = iter:max_epochs
    % Verbosity
    if verbose, fprintf('********** Epoch %i/%i (lr: %.0e, mom: %.2f, reg: %.0e) ********** \n', epoch, max_epochs, learning_rate, momentum, regularizer); end
    if visualize, waitbar(epoch/max_epochs, hwait, sprintf('Epoch %i/%i', epoch, max_epochs)); end
    
    % Shuffle inputs
    order = randperm(size(input,2));
    input = input(:,order);
    target = target(:,order);
    
    % Loop over batches
    lossbatch = zeros(1, Nbatch);
    gradbatch = zeros(1, Nbatch);
    chars = 0;
    for j=1:Nbatch
        % Verbosity
        if verbose
            for k=1:chars, fprintf('\b'), end;
            chars = fprintf('\tBatch %i/%i\n', j, Nbatch);
        end
        
        % Get batch data
        batch_input = input(:, batches{j});
        batch_target = target(:, batches{j});
        
        % Get current weights
        w = getwb(net);
        
        % Run minimization
        gradj = -backprop(net, batch_input, batch_target, 'Loss', loss); % Positive gradient
        if adadelta
            Egrad = momentum * Egrad + (1 - momentum) * (gradj') * gradj;
            Winc = -sqrt(Edelta + 1e-5) / sqrt(Egrad + 1e-5) * gradj;
            Edelta = momentum * Edelta + (1 - momentum) * (Winc') * Winc;
        else
            Winc = momentum * Winc - learning_rate * gradj - learning_rate * regularizer * w;
        end
        
        w = w + Winc;
        


        
        % Update weights
        net = setwb(net, w);
        
        % Compute error of mini-batch
%         lossbatch(j) = floss(batch_input, net(batch_input));
        lossbatch(j) = backprop_loss(batch_target, net(batch_input), loss);
        gradbatch(j) = norm(gradj);
        
        % Verbosity
%         if verbose
%             chars = chars + fprintf('\tLoss of mini-batch: %f\n', lossbatch(j));
%         end
%         
        % Visualization
        if visualize
            % h1 is shown below
            
            % Plot performance of current mini-batch
            plot(h2, 1:j, lossbatch(1:j), '-k')
            xlim(h2, [0.5 j+0.5])
            if j > 1, set(h2, 'xtick', [1 j]); end
            ylim(h2, [0 inf])
            xlabel(h2, 'Mini-batch');
            ylabel(h2, 'Performance')
            
            drawnow
        end
    end % End loop over batches
    
    % Update learning rate and momentum
%     learning_rate = learning_rate * learning_rate_mul;
%     momentum = momentum + momentum_inc;
    
    % Store performance parameters
    perf(epoch) = sum(lossbatch) / Nbatch;
    grad(epoch) = sum(gradbatch) / Nbatch;
    if Nval > 0, perf_val(epoch) = backprop_loss(val_target, net(val_input), loss); end
    
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
        semilogy(h1, 1:epoch, perf(1:epoch), '-*k', 'LineWidth', 1.5)
        if Nval > 0
            hold(h1, 'on')
            semilogy(h1, 1:epoch, perf_val(1:epoch), '-sr', 'LineWidth', 1.5)
            legend(h1, 'Training', 'Validation', 'Location', 'best')
            hold(h1, 'off')
        end
        xlim(h1, [0.5 epoch+0.5])
        if epoch > 1, set(h1, 'xtick', [1 epoch]); end
        ylim(h1, [0.5*min(perf(1:epoch)) inf])
        xlabel(h1, 'Epoch')
        ylabel(h1, 'Performance')
        if Nval > 0
            title(h1, sprintf('Best train/val performance: %f/%f', min(perf(1:epoch)), min(perf_val(1:epoch))))
        else
            title(h1, sprintf('Best performance: %f', min(perf(1:epoch))))
        end
            
        % Plot gradient norm of current epoch
        semilogy(h3, 1:epoch, grad(1:epoch), '-k')
        xlim(h3, [0.5 epoch+0.5])
        if epoch > 1, set(h3, 'xtick', [1 epoch]); end
        ylim(h3, [0.5*min(grad(1:epoch)) inf])
        xlabel(h3, 'Epoch');
        ylabel(h3, 'Gradient')
        
        % Update figures
        drawnow
    end
    
    % Save state
    if ~isempty(checkpoint_file)
        iter = epoch+1;
        if Nval > 0
            save(checkpoint_file, 'iter', 'w', 'Winc', 'perf', 'grad', 'perf_val');
        else
            save(checkpoint_file, 'iter', 'w', 'Winc', 'perf', 'grad');
        end
    end
    
    % Termination
    if grad(epoch) < min_grad
        fprintf('Gradient has reached a minimum at %f!', grad(epoch));
        fprintf('\tStopping backpropagation...\n');
        break
    end
    
%     % Termination
%     if Nval > 0 && epoch > 1
%         if perf_val(epoch) >= perf_val(epoch-1)
%             fprintf('Validation error has stagnated at %f!', perf_val(epoch));
%             if lr_dec < 3
%                 tmp = learning_rate / 10;
%                 fprintf('\tScaling learning rate: %f --> %f...\n', learning_rate, tmp);
%                 learning_rate = tmp;
%                 lr_dec = lr_dec + 1;
%             else
%                 fprintf('\tStopping backpropagation...\n');
%                 break
%             end
%         end
%     end

    if visualize
        if getappdata(hwait, 'canceling')
            fprintf('\tStopping backpropagation...\n');
            break
        end
    end
end % End epochs

if visualize, delete(hwait); end

