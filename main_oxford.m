%% Setup parameters for script

path('repeatibility', path);

% Set to true to disable loading existing autoencoders
force_training = false;

% Set to a positive value to reduce training set
Nreduce = 0;

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 200;
Niter_fine = 5 * Niter_init;

% Layer sizes
l1size = 1000;
l2size = 500;
l3size = 250;
l4size = 30;

%% Load data
% Use the helper functions to load the training images (column-major)
[~, train_images] = vl_ubcread_frames_descs('repeatibility/graf/img1.ppm.hesaff.patch', 'format', 'oxford');
train_images = double(train_images) / 255;

% Reduce training set
if Nreduce > 0
    warning('Reducing training set to %d examples...', Nreduce);
    train_images = train_images(:,1:Nreduce);
end

% Number of training/test cases
Ntrain = size(train_images,2);

% Ensure deterministic results
rng('default')

%% Load (or train) the autoencoders
if ~force_training &&...
        exist('data/patch_autoenc1.mat', 'file') &&...
        exist('data/patch_autoenc2.mat', 'file') &&...
        exist('data/patch_autoenc3.mat', 'file') &&...
        exist('data/patch_autoenc4.mat', 'file')
    disp 'Loading pretrained autoencoders from files...'
    load data/patch_autoenc1.mat
    load data/patch_autoenc2.mat
    load data/patch_autoenc3.mat
    load data/patch_autoenc4.mat
else
    % Training the first autoencoder

    % Set the size of the hidden layer for the autoencoder. For the autoencoder
    % that you are going to train, it is a good idea to make this smaller than
    % the input size.
    autoenc1 = trainAutoencoder(train_images,l1size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.15, ...
        'ScaleData', false);
    save('data/patch_autoenc1.mat', 'autoenc1');
    
    % More layers
    feat1 = encode(autoenc1,train_images);
    autoenc2 = trainAutoencoder(feat1,l2size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('data/patch_autoenc2.mat', 'autoenc2');

    feat2 = encode(autoenc2,feat1);
    autoenc3 = trainAutoencoder(feat2,l3size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('data/patch_autoenc3.mat', 'autoenc3');

    feat3 = encode(autoenc3,feat2);
    autoenc4 = trainAutoencoder(feat3,l4size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('data/patch_autoenc4.mat', 'autoenc4');
    
    clear feat1 feat2 feat3;
end

%% Stack the autoencoders
encoder = stack(autoenc1, autoenc2, autoenc3, autoenc4);
decoder = stack(get_decoder(autoenc4), get_decoder(autoenc3), get_decoder(autoenc2), get_decoder(autoenc1));
net = stack(encoder, decoder);
net.trainParam.epochs = Niter_fine;
net.trainParam.showWindow = true;

% Get features
enc_train_feat = encoder(train_images);

% Clearn up
clear autoenc1 autoenc2 autoenc3 autoenc4 encoder decoder;

%% Fine tune (or load fine tuned) network
if ~force_training && exist('data/patch_net_fine.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/patch_net_fine.mat;
else
    net_fine = train(net, train_images, train_images);
    save('data/patch_net_fine.mat', 'net_fine');
end

% Get encoder for fine tuned network
enc_fine = stack(get_layer(net_fine, 1), get_layer(net_fine,2), get_layer(net_fine, 3), get_layer(net_fine, 4));

%% Get a PCA for the training images
[c,~,~,~,~,mu] = pca(train_images', 'NumComponents', l4size);
pca_train_feat = (train_images'-repmat(mu,Ntrain,1)) * c;

%% Present reconstruction errors
% Reconstructions of training data before/after fine tuning and using PCA
net_train_rec = net(train_images);
net_fine_train_rec = net_fine(train_images);
pca_train_rec = pca_train_feat * c' + repmat(mu,Ntrain,1);
fprintf('PCA(%d) reconstruction error: %.4f\n', l4size, mse(pca_train_rec' - train_images));
fprintf('NN reconstruction error: %.4f\n', mse(net_train_rec - train_images));
fprintf('Fine-tuned NN reconstruction error: %.4f\n', mse(net_fine_train_rec - train_images));
idx = randi(Ntrain);
wh = sqrt(size(train_images,1)); % Image width/height
subplot(221),imagesc(reshape(train_images(:,idx), [wh wh])),title('Input image')
subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [wh wh])),title('PCA reconstruction')
subplot(223),imagesc(reshape(net_train_rec(:,idx), [wh wh])),title('NN reconstruction')
subplot(224),imagesc(reshape(net_fine_train_rec(:,idx), [wh wh])),title('Fine-tuned NN reconstruction')
colormap gray
