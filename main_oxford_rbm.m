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
num_hidden = [2000 1000 500 250 64];

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 100;

% Learning parameters
learning_rate = 0.01;
learning_rate_final = 0.001;
momentum = 0.5;
momentum_final = 0.9;

learning_rate_mul = exp(log(learning_rate_final / learning_rate) / Niter_fine);
momentum_inc = (momentum_final - momentum) / Niter_fine;

%% Load data
% Image root path
root='repeatibility/graf';

% Image(s) to consider
idxx = {'1', '2', '3', '4', '5', '6'};

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
% Our detectors: custom
detector = 'custom';
descriptor = 'patch';
binary = true; % Use with patch

% Use the helper functions to load the training images (column-major)
if ~(resume && exist('train_images', 'var') > 0)
    train_images = [];
    for i=1:numel(idxx)
        idx = idxx{i};
        pfile = [root '/img' idx '.ppm.' detector '.' descriptor];
        assert(exist(pfile, 'file') > 0);
        fprintf('Loading data from %s...\n', pfile);
        [~, tmp] = vl_ubcread_frames_descs(pfile, binary);
        train_images = [train_images tmp];
    end
    clear tmp;
end

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
if ~(resume && exist('batches', 'var') > 0)
    disp 'Creating batches...'
%     batches = create_batches(train_images', round(Ntrain/128), 'Method', 'ClusterPCA', 'Resize', 0.5, 'Verbose', true);
    batches = create_batches(train_images', round(Ntrain/128), 'Method', 'Random');
end

%% Train (or load) network
if resume && exist('data/oxford.mat', 'file')
    disp 'Loading fine tuned network file...'
    load data/oxford.mat;
else
    [net,enc,dec,enc_init,dec_init] = train_dbn(train_images', num_hidden,...
        'VisibleFunction', 'purelin',...
        'HiddenFunction', 'logsig',...
        'OutputFunction', 'purelin',...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', Niter_fine,...
        'BatchesInit', batches,...
        'Batches', batches,...
        'LearningRate', learning_rate,...
        'LearningRateMul', learning_rate_mul,...
        'Momentum', 0.5,...
        'MomentumInc', momentum_inc,...
        'Sigma', 0.01,...
        'Verbose', true,...
        'Visualize', true,...
        'Resume', resume);
    save('data/oxford.mat', 'net', 'enc', 'dec', 'enc_init', 'dec_init');
end

%% Network before fine tuning
net_init = stack(enc_init, dec_init);

%% Get a PCA for the training images
disp 'Getting a PCA...'
[c,mu] = train_pca(train_images', num_hidden(end));
pca_train_feat = project_pca(train_images', c, mu);

%% Present reconstruction errors
disp 'Presenting reconstruction results...'
% Reconstructions of training data before/after fine tuning and using PCA
pca_train_rec = reproject_pca(pca_train_feat, c, mu);
fprintf('    PCA(%d) reconstruction error: %.4f\n', num_hidden(end), mse(pca_train_rec' - train_images));
net_train_rec = net_init(train_images);
fprintf('    NN reconstruction error: %.4f\n', mse(net_train_rec - train_images));
net_fine_train_rec = net(train_images);
fprintf('    Fine-tuned NN reconstruction error: %.4f\n', mse(net_fine_train_rec - train_images));
idx = randi(Ntrain);
wh = sqrt(size(train_images,1)); % Image width/height
figure('Name', 'Example')
subplot(221),imagesc(reshape(train_images(:,idx), [wh wh])),title('Input image')
subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [wh wh])),title('PCA reconstruction')
subplot(223),imagesc(reshape(net_train_rec(:,idx), [wh wh])),title('NN reconstruction')
subplot(224),imagesc(reshape(net_fine_train_rec(:,idx), [wh wh])),title('Fine-tuned NN reconstruction')
colormap gray

%% Show some 1-layer unit weights
figure('Name', '1-layer encoder weights before fine tuning')
for i=1:100
    subplot(10,10,i),imagesc(reshape(enc_init.IW{1}(i,:)',wh,wh))
    axis off equal
end
colormap gray

figure('Name', '1-layer encoder weights after fine tuning')
for i=1:100
    subplot(10,10,i),imagesc(reshape(net.IW{1}(i,:)',wh,wh))
    axis off equal
end
colormap gray

disp 'All done!'