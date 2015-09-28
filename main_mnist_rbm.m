close all;

%% Setup parameters for script

% Set to true to disable loading existing autoencoders
force_training = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 200;

% Layer sizes
num_hidden = [1000 500 250 30];

%% Load data
% Use the helper functions to load the training/test images and labels
% (column-major)
[train_images, train_labels, test_images, test_labels] = load_mnist('mnist');

% Reduce training set
if Nreduce > 0
    warning('Reducing training set to %d examples...', Nreduce);
    train_images = train_images(:,1:Nreduce);
    train_labels = train_labels(1:Nreduce);
end

% Number of training/test cases
Ntrain = length(train_labels);
Ntest = length(test_labels);

% Ensure deterministic results
rng('default')

%% Fine tune (or load fine tuned) network
if ~force_training && exist('data/mnist_rbm_fine.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/mnist_rbm_fine.mat;
else
    [net,enc,dec,enc_init,dec_init] = train_dbn(train_images', num_hidden,...
        'OutputFunction', 'purelin',...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', Niter_fine,...
        'NumBatches', Ntrain/100,...
        'Verbose', true,...
        'Visualize', true);
    save('data/mnist_rbm_fine.mat', 'net', 'enc', 'dec', 'enc_init', 'dec_init');
end

% Network before fine tuning
net_init = stack(enc_init, dec_init);

%% Get a PCA for the training images
disp 'Getting a PCA...'
[c,~,~,~,~,mu] = pca(train_images', 'NumComponents', num_hidden(end));
pca_train_feat = (train_images'-repmat(mu,Ntrain,1)) * c;

%% Present reconstruction errors
disp 'Presenting results...'
% Reconstructions of training data before/after fine tuning and using PCA
pca_train_rec = pca_train_feat * c' + repmat(mu,Ntrain,1);
net_train_rec = net_init(train_images);
net_fine_train_rec = net(train_images);
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

%% Present classification results
% PCA
pca_test_feat = (test_images'-repmat(mu,Ntest,1)) * c;
model_knn_pca = fitcknn(pca_train_feat, train_labels, 'NumNeighbors', 5);
output_labels_pca = model_knn_pca.predict(pca_test_feat);
fprintf('PCA(%d) classification error rate: %.2f %%\n', l4size, 100 * sum(output_labels_pca ~= test_labels) / Ntest);

% Network
model_enc = fitcknn(enc_init(train_images)', train_labels, 'NumNeighbors', 5);
output_labels_enc = model_enc.predict(enc_init(test_images)');
fprintf('NN error rate: %.2f %%\n', 100 * sum(output_labels_enc ~= test_labels) / Ntest);

% Fine tuned network
model_encfine = fitcknn(enc(train_images)', train_labels, 'NumNeighbors', 5);
output_labels_encfine = model_encfine.predict(enc(test_images)');
fprintf('Fine-tuned NN error rate: %.2f %%\n', 100 * sum(output_labels_encfine ~= test_labels) / Ntest);

%% Show some 1-layer unit weights
figure('Name', '1-layer encoder weights')
for i=1:100
    subplot(10,10,i),imagesc(reshape(net.IW{1}(i,:)',wh,wh))
end
colormap gray
