% This example creates a stack of four autoencoders on a subset of the
% MNIST dataset

%% Setup parameters for script

% Set to true to disable loading existing autoencoders
force_training = true;

% Set to a positive value to reduce training set
Nreduce = 10000;

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
% Use the helper functions to load the training/test images and labels
% (column-major)
[train_images, train_labels, test_images, test_labels] = load_mnist;
size(train_images)
size(train_labels)

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

if ~force_training &&...
        exist('autoenc1.mat', 'file') &&...
        exist('autoenc2.mat', 'file') &&...
        exist('autoenc3.mat', 'file') &&...
        exist('autoenc4.mat', 'file')
    disp 'Loading pretrained autoencoders from files...'
    load autoenc1.mat
    load autoenc2.mat
    load autoenc3.mat
    load autoenc4.mat
else
    %% Training the first autoencoder

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
    save('autoenc1.mat', 'autoenc1');

    %% Training the second autoencoder
    feat1 = encode(autoenc1,train_images);
    autoenc2 = trainAutoencoder(feat1,l2size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('autoenc2.mat', 'autoenc2');

    %% More layers
    feat2 = encode(autoenc2,feat1);
    autoenc3 = trainAutoencoder(feat2,l3size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('autoenc3.mat', 'autoenc3');

    feat3 = encode(autoenc3,feat2);
    autoenc4 = trainAutoencoder(feat3,l4size, ...
        'MaxEpochs',Niter_init, ...
        'DecoderTransferFunction','logsig',...
        'L2WeightRegularization',0.002, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.1, ...
        'ScaleData', false);
    save('autoenc4.mat', 'autoenc4');
end

%% Stack the autoencoders
encoder = stack(autoenc1, autoenc2, autoenc3, autoenc4);
decoder = stack(get_decoder(autoenc4), get_decoder(autoenc3), get_decoder(autoenc2), get_decoder(autoenc1));
net = stack(encoder, decoder);

% Get features
enc_train_feat = encoder(train_images);
enc_test_feat = encoder(test_images);

%% Fine-tune the whole network
net.trainParam.epochs = Niter_fine;
net.trainParam.showWindow = true;
net_fine = net;

% ATTENTION: All code below this point can be re-run to further fine tine the net
net_fine = train(net_fine, train_images, train_images);
save('net_fine.mat', 'net_fine');

% Get efor fineworked network
enc_fine = stack(get_layer(net_fine, 1), get_layer(net_fine,2), get_layer(net_fine, 3), get_layer(net_fine, 4));

%% Get a PCA for the training images
[c,~,~,~,~,mu] = pca(train_images', 'NumComponents', l4size);
pca_train_feat = (train_images'-repmat(mu,Ntrain,1)) * c;

%% Present reconstruction errors
% Reconstructions of training data before/after fine tuning and using PCA
net_train_rec = net(train_images);
net_fine_train_rec = net_fine(train_images);
pca_train_rec = pca_train_feat * c' + repmat(mu,Ntrain,1);
fprintf('PCA(%d) reconstruction error: %.2e\n', l4size, sum(abs(pca_train_rec(:) - train_images(:))));
fprintf('NN reconstruction error: %.2e\n', sum(abs(net_train_rec(:) - train_images(:))));
fprintf('Fine-tuned NN reconstruction error: %.2e\n', sum(abs(net_fine_train_rec(:) - train_images(:))));
idx = randi(Ntrain);
subplot(221),imagesc(reshape(train_images(:,idx), [28 28])),title('Input image')
subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [28 28])),title('PCA reconstruction')
subplot(223),imagesc(reshape(net_train_rec(:,idx), [28 28])),title('NN reconstruction')
subplot(224),imagesc(reshape(net_fine_train_rec(:,idx), [28 28])),title('Fine-tuned NN reconstruction')
colormap gray

%% Present classification results
% PCA
pca_test_feat = (test_images'-repmat(mu,Ntest,1)) * c;
model_knn_pca = fitcknn(pca_train_feat, train_labels, 'NumNeighbors', 5);
output_labels_pca = model_knn_pca.predict(pca_test_feat);
fprintf('PCA(%d) classification rrror rate: %.2f %%\n', l4size, 100 * sum(output_labels_pca ~= test_labels) / Ntest);

% Network
model_enc = fitcknn(enc_train_feat', train_labels, 'NumNeighbors', 5);
output_labels_enc = model_enc.predict(enc_test_feat');
fprintf('NN error rate: %.2f %%\n', 100 * sum(output_labels_enc ~= test_labels) / Ntest);

% Fine tuned network
model_encfine = fitcknn(enc_fine(train_images)', train_labels, 'NumNeighbors', 5);
output_labels_encfine = model_encfine.predict(enc_fine(test_images)');
fprintf('Fine-tuned NN error rate: %.2f %%\n', 100 * sum(output_labels_encfine ~= test_labels) / Ntest);

% % Also for Hinton's DBN
% load mnist_weights
% fdbn_train = prop(train_images', w1, w2, w3, w4);
% fdbn_test = prop(test_images', w1, w2, w3, w4);
% model_knn_dbn = fitcknn(fdbn_train, train_labels, 'NumNeighbors', 5);
% output_labels_dbn = model_knn_dbn.predict(fdbn_test);
% disp 'Presenting DBN results...'
% fprintf('\tError rate: %.2f %%\n', 100 * sum(output_labels_dbn ~= test_labels') / Ntest);
