%% Setup parameters for script

path('repeatibility', path);

% Set to true to disable loading existing autoencoders
force_training = false;

% Set to a positive value to reduce training set
Nreduce = 0;

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 200;

% Layer sizes
l1size = 2000;
l2size = 1000;
l3size = 250;
l4size = 30;

%% Load data
% Image root path
root='repeatibility/graf';

% Image(s) to consider
idxx = {'1'};%, '2', '3', '4', '5', '6'};

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
detector='hesaff';

% Use the helper functions to load the training images (column-major)
train_images = [];
for i=1:numel(idxx)
    idx = idxx{i};
    pfile = [root '/img' idx '.ppm.' detector '.patch'];
    assert(exist(pfile, 'file') > 0);
    [~, tmp] = vl_ubcread_frames_descs('repeatibility/graf/img1.ppm.hesaff.patch');
    train_images = [train_images tmp];
end

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
        exist('data/patch_rbm1.mat', 'file') &&...
        exist('data/patch_rbm2.mat', 'file') &&...
        exist('data/patch_rbm3.mat', 'file') &&...
        exist('data/patch_rbm4.mat', 'file')
    disp 'Loading pretrained autoencoders from files...'
    load data/patch_rbm1.mat
    load data/patch_rbm2.mat
    load data/patch_rbm3.mat
    load data/patch_rbm4.mat
else
    [enc1,dec1] = train_rbm(train_images, l1size,...
        'MaxEpochs', Niter_init,...
        'NumBatches', Ntrain/10,...
        'RowMajor', false,...
        'Verbose', true,...
        'Visualize', true);
    rbm1 = stack(enc1,dec1);
    save('data/patch_rbm1.mat', 'rbm1');
    
    feat1 = enc1(train_images);
    [enc2,dec2] = train_rbm(feat1, l2size,...
        'MaxEpochs', Niter_init,...
        'NumBatches', Ntrain/10,...
        'RowMajor', false,...
        'Verbose', true,...
        'Visualize', false);
    rbm2 = stack(enc2,dec2);
    save('data/patch_rbm2.mat', 'rbm2');
    
    feat2 = enc2(feat1);
    [enc3,dec3] = train_rbm(feat2, l3size,...
        'MaxEpochs', Niter_init,...
        'NumBatches', Ntrain/10,...
        'RowMajor', false,...
        'Verbose', true,...
        'Visualize', false);
    rbm3 = stack(enc3,dec3);
    save('data/patch_rbm3.mat', 'rbm3');
    
    feat3 = enc3(feat2);
    [enc4,dec4] = train_rbm(feat3, l4size,...
        'MaxEpochs', Niter_init,...
        'NumBatches', Ntrain/10,...
        'RowMajor', false,...
        'Verbose', true,...
        'Visualize', false);
    rbm4 = stack(enc4,dec4);
    save('data/patch_rbm4.mat', 'rbm4');
end

%% Stack the RBMs
encoder = stack(get_layer(rbm1,1), get_layer(rbm2,1), get_layer(rbm3,1), get_layer(rbm4,1));
decoder = stack(get_layer(rbm4,2), get_layer(rbm3,2), get_layer(rbm2,2), get_layer(rbm1,2));
net = stack(encoder, decoder);
net.trainParam.epochs = Niter_fine;
net.trainParam.showWindow = true;
net.divideFcn = 'dividetrain';
net.plotFcns = {'plotperform'};
net.plotParams = {nnetParam}; % Dummy?

% Get features
enc_train_feat = encoder(train_images);

%% Fine tune (or load fine tuned) network
if ~force_training && exist('data/patch_rbm_fine.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/patch_rbm_fine.mat;
else
    net_fine = train(net, train_images, train_images);
    save('data/patch_rbm_fine.mat', 'net_fine');
end

% Get encoder for fine tuned network
enc_fine = stack(get_layer(net_fine, 1), get_layer(net_fine,2), get_layer(net_fine, 3), get_layer(net_fine, 4));

%% Get a PCA for the training images
disp 'Getting a PCA...'
[c,~,~,~,~,mu] = pca(train_images', 'NumComponents', l4size);
pca_train_feat = (train_images'-repmat(mu,Ntrain,1)) * c;

%% Present reconstruction errors
disp 'Presenting results...'
% Reconstructions of training data before/after fine tuning and using PCA
net_train_rec = net(train_images);
net_fine_train_rec = net_fine(train_images);
pca_train_rec = pca_train_feat * c' + repmat(mu,Ntrain,1);
fprintf('PCA(%d) reconstruction error: %.4f\n', l4size, mse(pca_train_rec' - train_images));
fprintf('NN reconstruction error: %.4f\n', mse(net_train_rec - train_images));
fprintf('Fine-tuned NN reconstruction error: %.4f\n', mse(net_fine_train_rec - train_images));
idx = randi(Ntrain);
wh = sqrt(size(train_images,1)); % Image width/height
figure('Name', 'Example')
subplot(221),imagesc(reshape(train_images(:,idx), [wh wh])),title('Input image')
subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [wh wh])),title('PCA reconstruction')
subplot(223),imagesc(reshape(net_train_rec(:,idx), [wh wh])),title('NN reconstruction')
subplot(224),imagesc(reshape(net_fine_train_rec(:,idx), [wh wh])),title('Fine-tuned NN reconstruction')
colormap gray

%% Show some 1-layer unit weights
figure('Name', '1-layer encoder weights')
for i=1:100
    subplot(10,10,i),imagesc(reshape(net_fine.IW{1}(1,:)',wh,wh))
end
colormap gray
