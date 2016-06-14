close all;

%% Setup parameters for script

% Ensure deterministic results
rng('default')

% Set to true to enable re-use of training data and networks
resume = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Layer sizes
num_hidden = 1000;

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 150;
Niter_fine = 150;
classify = true; % Set to true to avoid backprop on autoencoder stack

% Learning parameters
learning_rate = 0.01;
momentum = 0.9;

gauss_noise = 0;
mask_noise = 0.5;
regularizer_init = 0;
dropout_rate_init = 0;

regularizer = 0.5;
dropout_rate = 0.5;

in_hid_out = {'logsig', 'logsig', 'logsig'};
% in_hid_out = {'logsig', 'logsig', 'purelin'};
% in_hid_out = {'tansig', 'tansig', 'tansig'};
% in_hid_out = {'poslin', 'poslin', 'purelin'};
loss = 'binary_crossentropy';
% loss = 'mse';

%% Load data
% Use the helper functions to load the training/test images and labels
% (column-major)
[train_images, train_labels, test_images, test_labels] = load_mnist('mnist');
% Number of training/test cases
Ntrain = length(train_labels);
Ntest = length(test_labels);
% Encode for classification also
[classes,~,ictrain] = unique(train_labels);
[~,~,ictest] = unique(test_labels);
Ttrain = zeros(length(classes), Ntrain);
Ttest = zeros(length(classes), Ntest);
for i=1:Ntrain, Ttrain(ictrain(i), i) = 1; end
for i=1:Ntest, Ttest(ictest(i), i) = 1; end

%% Reduce training set
if Nreduce > 0
    idx = randperm(Ntrain);
    idx = idx(1:Nreduce);
    warning('Reducing training set to %d examples...', Nreduce);
    train_images = train_images(:,idx);
    train_labels = train_labels(idx);
    Ttrain = Ttrain(:,idx);
    Ntrain = length(train_labels);
end

%% Create batches
if ~(resume && exist('batches_init', 'var') > 0 && exist('batches', 'var') > 0)
    disp 'Creating batches...'
    batches_init = create_batches(train_images', round(Ntrain/100), 'Method', 'Random');
%     batches = create_batches(train_images', round(Ntrain/1000), 'Method', 'Random');
    batches = batches_init;
end

%% Train (or load) network
if resume && exist('data/mnist_ae.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/mnist_ae.mat;
else
    disp 'Starting pretraining...'
    [net, net_init] = train_sae(train_images', num_hidden,...
        'InputFunction', in_hid_out{1},...
        'HiddenFunction', in_hid_out{2},...
        'OutputFunction', in_hid_out{3},...
        'TiedWeights', true,...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', ~classify*Niter_fine,...
        'Loss', loss,...
        'BatchesInit', batches_init,...
        'Batches', batches,...
        'ValidationFraction', 0,...
        'MaskingNoise', mask_noise,...
        'GaussianNoise', gauss_noise,...
        'DropoutRate', dropout_rate_init,...
        'LearningRate', learning_rate,...
        'Momentum', momentum,...
        'Regularizer', regularizer_init,...
        'Width', 28,...
        'Verbose', true,...
        'Visualize', true,...
        'UseGPU', true,...
        'Resume', true);
    delete 'net.mat';
    if ~exist('data', 'dir'), mkdir data; end
    save('data/mnist_ae.mat', 'net', 'net_init');
end

% Get encoder for initial/final network
enc = get_layer(net,1);
enc_init = get_layer(net_init,1);
for i=2:length(num_hidden)
    enc = stack(enc, get_layer(net,i));
    enc_init = stack(enc_init, get_layer(net_init,i));
end

if classify
    %% Train a softmax classifier using the initial network
    chkpnt = 'data/soft.mat';
    lsoft = create_layer(enc.outputs{end}.size, 10, 'softmax', 0.01*randn(10,enc.outputs{end}.size), 0.01*randn(10,1), 'trainscg');
    lsoft.performFcn = 'crossentropy';
    soft = stack(enc, lsoft);
    soft.divideFcn = 'dividetrain';
    soft.trainParam.epochs = Niter_fine;
    soft.trainParam.min_grad = 1e-15;
    soft = sgd(soft, train_images, Ttrain, Niter_fine, learning_rate,...
        'Batches', batches,...
        'ValidationFraction', 0.1,...
        'Loss', 'crossentropy',...
        'Optimizer', 'adam',...
        'Momentum', momentum,...
        'Regularizer', regularizer,...
        'DropoutRate', dropout_rate,...
        'Verbose', true,...
        'TestData', struct('input', test_images, 'target', Ttest),...
        'Visualize', true,...
        'CheckpointFile', chkpnt,...
        'UseGPU', false);


    %% Present classification results
    % TODO: Standardize the test set, right now we get misleading results
    figure,plotconfusion(Ttest, soft(test_images))
else
    if num_hidden(end) < size(train_images,1)
        %% Get a PCA for the training images
        disp 'Getting a PCA...'
        [c,mu] = train_pca(train_images', num_hidden(end));
        pca_train_feat = project_pca(train_images', c, mu);

        %% Present reconstruction errors
        disp 'Presenting reconstruction errors (cross entropy)...'
        % Reconstructions of training data before/after fine tuning and using PCA
        floss = @(t,y) sum(sum( -t .* log(y + eps) - (1 - t) .* log(1 - y + eps) )) / numel(t);
        pca_train_rec = reproject_pca(pca_train_feat, c, mu);
        fprintf('    PCA(%d) reconstruction error: %.4f\n', num_hidden(end), floss(train_images,pca_train_rec'));
        net_train_rec = net_init(train_images);
        fprintf('    NN reconstruction error: %.4f\n', floss(train_images,net_train_rec));
        net_fine_train_rec = net(train_images);
        fprintf('    Fine-tuned NN reconstruction error: %.4f\n', floss(train_images,net_fine_train_rec));
        idx = randi(Ntrain);
        wh = sqrt(size(train_images,1)); % Image width/height
        figure('Name', 'Example')
        subplot(221),imagesc(reshape(train_images(:,idx), [wh wh])),title('Input image')
        subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [wh wh])),title('PCA reconstruction')
        subplot(223),imagesc(reshape(net_train_rec(:,idx), [wh wh])),title('NN reconstruction')
        subplot(224),imagesc(reshape(net_fine_train_rec(:,idx), [wh wh])),title('Fine-tuned NN reconstruction')
        colormap gray
    end
end

%% Show some 1-layer unit weights
wh = sqrt(size(train_images,1)); % Image width/height
figure('Name', '1-layer encoder weights before fine tuning')
for i=1:100
    subtightplot(10,10,i),imagesc(reshape(net_init.IW{1}(i,:)',wh,wh))
    axis off equal
end
colormap gray

figure('Name', '1-layer encoder weights after fine tuning')
for i=1:100
    subtightplot(10,10,i),imagesc(reshape(net.IW{1}(i,:)',wh,wh))
    axis off equal
end
colormap gray

disp 'All done!'
