close all;

%% Setup parameters for script

% Ensure deterministic results
rng('default')

% Set to true to enable re-use of training data and networks
resume = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Layer sizes
% num_hidden = [1000 500 250 30];
num_hidden = [500 500 2000];
% num_hidden = 1000;

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 0;

% Learning parameters
learning_rate = 0.05;
learning_rate_final = 0.0005;
momentum = 0.95;
momentum_final = 0.9;

learning_rate_mul = exp(log(learning_rate_final / learning_rate) / Niter_fine);
momentum_inc = (momentum_final - momentum) / Niter_fine;

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
    Ntrain = length(train_labels);
end

%% Create batches
if ~(resume && exist('batches_init', 'var') > 0 && exist('batches', 'var') > 0)
    disp 'Creating batches...'
    batches_init = create_batches(train_images', round(Ntrain/100), 'Method', 'Random');
    batches = create_batches(train_images', round(Ntrain/1000), 'Method', 'Random');
end

%% Train (or load) network
if resume && exist('data/mnist_ae.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/mnist_ae.mat;
else
    [net, net_init] = train_sae(train_images', num_hidden,...
        'InputFunction', 'logsig',...
        'HiddenFunction', 'logsig',...
        'OutputFunction', 'logsig',...
        'TiedWeights', true,...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', Niter_fine,...
        'Loss', 'binary_crossentropy',...
        'BatchesInit', batches_init,...
        'Batches', batches,...
        'ValidationFraction', 0,...
        'MaskingNoise', 0.5,...
        'LearningRate', learning_rate,...
        'LearningRateMul', learning_rate_mul,...
        'Momentum', momentum,...
        'Regularizer', 0,...
        'Sigma', 0.1,...
        'Width', 28,...
        'Verbose', true,...
        'Visualize', true,...
        'UseGPU', true,...
        'Resume', true);
    save('data/mnist_ae.mat', 'net', 'net_init');
end
    
% Get encoder for initial/final network
enc = get_layer(net,1);
enc_init = get_layer(net_init,1);
for i=2:length(num_hidden)
    enc = stack(enc, get_layer(net,i));
    enc_init = stack(enc_init, get_layer(net_init,i));
end

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

%% Train a softmax classifier using the initial network
lsoft = create_layer(enc.outputs{end}.size, 10, 'softmax', 0.1*rand(10,enc.outputs{end}.size), 0.1*rand(10,1), 'traincgp');
lsoft.performFcn = 'crossentropy';
soft = stack(enc, lsoft);
soft.divideFcn = 'dividetrain';
% soft = train(soft, train_images, Ttrain, 'CheckpointFile', 'soft');
soft = sgd(soft, train_images, Ttrain, Niter_fine, learning_rate,...
    'Batches', batches,...
    'ValidationFraction', 0.1,...
    'Loss', 'crossentropy',...
    'Adadelta', true,...
    'Momentum', momentum,...
    'Regularizer', 0,...
    'Verbose', true,...
    'Visualize', true,...
    'CheckpointFile', 'sgd_soft.mat');


%% Present classification results
figure,plotconfusion(Ttest, soft(test_images))

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
