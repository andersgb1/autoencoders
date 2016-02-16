close all;

%% Setup parameters for script

% Ensure deterministic results
rng('default')

path('repeatibility', path);

% Set to true to enable re-use of training data and networks
resume = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Layer sizes
num_hidden = [2000];

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 250;

% Learning parameters
learning_rate = 0.5;
learning_rate_final = 0.0001;
momentum = 0.95;

learning_rate_mul = exp(log(learning_rate_final / learning_rate) / Niter_fine);

%% Load data% Data root path
root = 'shape';
% shape_file = 'bologna_shot.txt';
% shape_file = 'bologna_si153.txt';
% width = 9;
% shape_file = 'bologna_si861.txt';
shape_file = 'all_si861.txt';
width = 21;

% Load training data
train_images = dlmread([root '/' shape_file])';

% Number of training cases
Ntrain = size(train_images,2);

%% Reduce training set
if Nreduce > 0
    idx = randperm(Ntrain);
    idx = idx(1:Nreduce);
    warning('Reducing training set to %d examples...', length(idx));
    train_images = train_images(:,idx);
    Ntrain = size(train_images,2);
end

%% Create batches
if ~(resume && exist('batches_init', 'var') > 0 && exist('batches', 'var') > 0)
    disp 'Creating batches...'
%     batches = create_batches(train_images', round(Ntrain/128), 'Method', 'ClusterPCA', 'Resize', 0.5, 'Verbose', true);
    batches_init = create_batches(train_images', round(Ntrain/128), 'Method', 'Random');
    batches = batches_init;%create_batches(train_images', round(Ntrain/1000), 'Method', 'Random');
end

%% Train (or load) network
if resume && exist('data/shape.mat', 'file')
    disp 'Loading fine tuned network file...'
    load data/shape.mat;
else
    [net, net_init] = train_sae(train_images', num_hidden,...
        'InputFunction', 'purelin',...
        'HiddenFunction', 'logsig',...
        'OutputFunction', 'purelin',...
        'TiedWeights', true,...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', Niter_fine,...
        'Loss', 'mse',...
        'BatchesInit', batches_init,...
        'Batches', batches,...
        'ValidationFraction', 0,...
        'GaussianNoise', 0.5,...
        'MaskingNoise', 0,...
        'LearningRate', learning_rate,...
        'LearningRateMul', learning_rate_mul,...
        'Momentum', momentum,...
        'Regularizer', 0,...
        'Sigma', 0.1,...
        'Width', width,...
        'Verbose', true,...
        'Visualize', true,...
        'UseGPU', true,...
        'Resume', true);
    save('data/shape.mat', 'net', 'net_init');
end

%% Get a PCA for the training images
if num_hidden(end) < size(train_images,1)
    disp 'Getting a PCA...'
    [c_pca,mu_pca] = train_pca(train_images', num_hidden(end));
    pca_train_feat = project_pca(train_images', c_pca, mu_pca);
    
    %% Present reconstruction errors
    disp 'Presenting reconstruction results...'
    % Reconstructions of training data before/after fine tuning and using PCA
    pca_train_rec = reproject_pca(pca_train_feat, c_pca, mu_pca);
    fprintf('    PCA(%d) reconstruction error: %f\n', num_hidden(end), mse(pca_train_rec' - train_images));
    % TODO
    train_images_std = ( (train_images' - repmat(mu, Ntrain, 1)) / sigma )';
    net_train_rec = net_init(train_images_std);
    fprintf('    NN reconstruction error: %f\n', mse(net_train_rec*sigma+repmat(mu',1,Ntrain) - train_images));
    net_fine_train_rec = net(train_images_std);
    fprintf('    Fine-tuned NN reconstruction error: %f\n', mse(net_fine_train_rec*sigma+repmat(mu',1,Ntrain) - train_images));
end

disp 'All done!'
