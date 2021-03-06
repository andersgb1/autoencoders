close all;

%% Setup parameters for script

% Ensure deterministic results
rng('default')

% Set to true to enable re-use of training data and networks
resume = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Layer sizes
num_hidden = [1000 500 250 30];

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 50;
Niter_fine = 50;

% Learning parameters
learning_rate = 0.1;
learning_rate_final = 0.001;
momentum = 0.5;
momentum_final = 0.9;

learning_rate_mul = exp(log(learning_rate_final / learning_rate) / Niter_fine);
momentum_inc = (momentum_final - momentum) / Niter_fine;

%% Load data
% Use the helper functions to load the training/test images and labels
% (column-major)
[train_images, train_labels, test_images, test_labels] = load_mnist('mnist');

%% Reduce training set
% Number of training/test cases
Ntrain = length(train_labels);
Ntest = length(test_labels);

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
if resume && exist('data/mnist.mat', 'file')
    disp 'Loading pretrained fine tuned network file...'
    load data/mnist.mat;
else
    [net,enc,dec,enc_init,dec_init] = train_dbn(train_images', num_hidden,...
        'VisibleFunction', 'logsig',...
        'HiddenFunction', 'logsig',...
        'OutputFunction', 'purelin',...
        'MaxEpochsInit', Niter_init,...
        'MaxEpochs', Niter_fine,...
        'BatchesInit', batches_init,...
        'Batches', batches,...
        'LearningRate', learning_rate,...
        'LearningRateMul', learning_rate_mul,...
        'Momentum', 0.5,...
        'MomentumInc', momentum_inc,...
        'Sigma', 0.1,...
        'Verbose', true,...
        'Visualize', true,...
        'Resume', resume);
    save('data/mnist.mat', 'net', 'enc', 'dec', 'enc_init', 'dec_init');
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

%% Present classification results
disp 'Presenting classification results...'
k=1;
% PCA
disp '    Training NN classifier using PCA...'
pca_test_feat = (test_images'-repmat(mu,Ntest,1)) * c;
disp '    Testing...'
model_knn_pca = fitcknn(pca_train_feat, train_labels, 'NumNeighbors', k);
output_labels_pca = model_knn_pca.predict(pca_test_feat);
fprintf('    PCA(%d) classification error rate: %.2f %%\n', num_hidden(end), 100 * sum(output_labels_pca ~= test_labels) / Ntest);

% Network
disp '    Training NN classifier using initial network...'
model_enc = fitcknn(enc_init(train_images)', train_labels, 'NumNeighbors', k);
disp '    Testing...'
output_labels_enc = model_enc.predict(enc_init(test_images)');
fprintf('    NN error rate: %.2f %%\n', 100 * sum(output_labels_enc ~= test_labels) / Ntest);

% Fine tuned network
disp '    Training NN classifier using fine tuned network...'
model_encfine = fitcknn(enc(train_images)', train_labels, 'NumNeighbors', k);
disp '    Testing...'
output_labels_encfine = model_encfine.predict(enc(test_images)');
fprintf('    Fine-tuned NN error rate: %.2f %%\n', 100 * sum(output_labels_encfine ~= test_labels) / Ntest);

%% Show some 1-layer unit weights
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
