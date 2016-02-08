function net = sgd(net, input, target, max_epochs, learning_rate, varargin)

%% Get opts
if nargin < 4, max_epochs = 100; end
if nargin < 5, learning_rate = 1; end
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Batches', {}, @iscell)
p.addParameter('ValidationFraction', 0.1, @isnumeric)
p.addParameter('Loss', 'crossentropy', @ischar)
p.addParameter('Momentum', 0, @isnumeric)
p.addParameter('Regularizer', 0, @isnumeric)
p.addParameter('Verbose', false, @islogical)
p.addParameter('Visualize', false, @islogical)
p.parse(varargin{:});
batches = p.Results.Batches;
val_frac = p.Results.ValidationFraction;
assert(val_frac >= 0 && val_frac < 1, 'Validation fraction must be a number in [0,1[!')
loss = p.Results.Loss;
momentum = p.Results.Momentum;
regularizer = p.Results.Regularizer;
verbose = p.Results.Verbose;
visualize = p.Results.Visualize;

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
    h1 = subplot(1,2,1);
    h2 = subplot(1,2,2);
end

%% Start backpropagation
lr_dec = 0; % Number of times we decreased the learning rates
perf = zeros(1, max_epochs);
Winc = 0;
for epoch = 1:max_epochs
    % Verbosity
    if verbose, fprintf('********** Epoch %i/%i (lr: %.0e, mom: %.2f, reg: %.0e) ********** \n', epoch, max_epochs, learning_rate, momentum, regularizer); end
    
    % Shuffle inputs
    order = randperm(size(input,2));
    input = input(:,order);
    target = target(:,order);
    
    % Loop over batches
    lossbatch = zeros(1, Nbatch);
    for j=1:Nbatch
        % Verbosity
        if verbose, chars = fprintf('\tBatch %i/%i\n', j, Nbatch); end
        
        % Get batch data
        batch_input = input(:, batches{j});
        batch_target = target(:, batches{j});
        
        % Get current weights
        w = getwb(net);
        
        % Run minimization
        gradj = -backprop(net, batch_input, batch_target, 'Loss', loss);
        Winc = momentum * Winc - learning_rate * gradj - learning_rate * regularizer * w;
        w = w + Winc;
        
        % Update weights
        net = setwb(net, w);
        
        % Compute error of mini-batch
%         lossbatch(j) = floss(batch_input, net(batch_input));
        lossbatch(j) = backprop_loss(batch_target, net(batch_input), loss);
        
        % Verbosity
        if verbose
            chars = chars + fprintf('\tLoss of mini-batch: %f\n', lossbatch(j));
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
            ylabel(h2, 'Performance')
            drawnow
        end
    end % End loop over batches
    
    % Update learning rate and momentum
%     learning_rate = learning_rate * learning_rate_mul;
%     momentum = momentum + momentum_inc;
    
    % Store performance
    perf(epoch) = sum(lossbatch) / Nbatch;
%     if Nval > 0, perf_val(epoch) = floss(input_val', net(input_val')); end
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
        drawnow
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
end % End epochs
