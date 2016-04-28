close all;

%% Setup parameters for script

% Ensure deterministic results
rng('default')

path('repeatibility', path);

% Set to true to enable re-use of training data and networks
resume = true;

% Set to a positive value to reduce training set
Nreduce = 0;

% Set to true to apply data augmentation
augment = false;

% Layer sizes
num_hidden = [4000 2000 1000 500 250];

% Number of training iterations for the individual layers and for the final
% fine tuning
Niter_init = 1000*ones(1,length(num_hidden));
Niter_fine = 13;

% Learning parameters
learning_rate = 0.1;
% learning_rate_final = 0.0001;
momentum = 0.9;
momentum_final = 0.9;

learning_rate_mul = 1;%exp(log(learning_rate_final / learning_rate) / Niter_fine);
momentum_inc = (momentum_final - momentum) / Niter_fine;

%% Load data
% Image root path
root='cifar-10-batches-mat';

% Use the helper functions to load the training images (column-major)
if ~(resume && exist('train_images', 'var') > 0)
    trainfile_resized = sprintf('%s/data_batch_all_resized.mat', root);
    if exist(trainfile_resized, 'file') > 0
        load(trainfile_resized);
    else
        % Load original CIFAR training batches
        train_images = [];
        for i=1:5
            load(sprintf('%s/data_batch_%i.mat', root, i))
            train_images = [train_images data']; % Column-major, scaled to [0,1]
        end
        % Convert to grayscale, upscale to 41x41 px
        data = zeros(41*41, size(train_images,2));
        disp 'Resizing images 32x32 --> 41x41 px...'
        chars = 0;
        for i=1:size(train_images, 2)
            for j = 1:chars, fprintf('\b'); end
            chars = fprintf('%i/%i\n', i, size(train_images,2));
            img = reshape(train_images(:,i), 32, 32, 3);
            img = rgb2gray(img);
            img = imresize(img, [41 41]);
            img = double(img) / 255;
            data(:,i) = reshape(img, 41*41, 1);
        end
        train_images = data;
        clear data;
        % Save resized images
        save(trainfile_resized, 'train_images')
    end
end

if augment
    trainfile_augmented = sprintf('%s/data_batch_all_resized_augmented.mat', root);
    if resume && exist(trainfile_augmented, 'file') > 0
        warning('Loading augmented training set!')
        load(trainfile_augmented)
    else
        warning('Artificially changing the training set by augmentation!')
        [dim, N] = size(train_images);
        Naugment = 5000;
        % Reduce to Naugment examples
        idx_reduce = randperm(N);
        train_images = train_images(:, idx_reduce(1:Naugment));
        N = size(train_images, 2);
        
        % Rotation
%         rotations = 0:90:270;
        rotations = 0;
        Nrot = length(rotations);
        % Create background masks for rotated images
        Mrots = false(41, 41, Nrot);
        for j=1:Nrot
            Mrots(:,:,j) = ~imrotate(true(41, 41), rotations(j), 'crop');
            Mrots(:,:,j) = Mrots(:,:,j) & ~imclearborder(Mrots(:,:,j));
        end
        tmp = zeros(dim, Nrot * N);
        idxtmp = 0;
        for i=1:N
            imgi = train_images(:,i);
%             mu = mean(imgi); % Mean intensity of training case
            % As 2D image
            imgi = reshape(imgi, 41, 41);
            % Rotate
            for j=1:Nrot
                % Rotate
                imrot = imrotate(imgi, rotations(j), 'crop');
                 % Set background to mean intensity
                mask = Mrots(:,:,j);
%                 imrot(mask) = mu; % Set background to mean
                imrot(mask) = 0; % Set background to black
                % As vector
                tmp(:,idxtmp+j) = reshape(imrot, dim, 1);
            end
            idxtmp = idxtmp + Nrot;
        end
        train_images = tmp;
        N = size(train_images, 2);
        
        % Gamma
        gammas = 1 ./ [0.5 0.4 0.3 0.2 0.1];
%         gammas = 1;
        Ngamma = length(gammas);
        tmp = zeros(dim, Ngamma * N);
        idxtmp = 0;
        for i=1:N
            imgi = train_images(:,i);
%             tmp(:,idxtmp) = imgi; % Original
            % Apply gamma
            for j=1:Ngamma
                tmp(:,idxtmp+j) = imgi.^gammas(j);
            end
            idxtmp = idxtmp + Ngamma;
        end
        train_images = tmp;
        N = size(train_images, 2);
        
        % Scaling
%         scales = [1 1.4 1.6 1.8];
        scales = 1;
        Nscale = length(scales);
        % Prepare crop masks
        crops = zeros(Nscale, 4);
        for j=1:Nscale
            newsz = scales(j) * 41;
            szdiff = newsz - 41;
            offset = floor(szdiff / 2);
            crops(j,:) = [offset+1 offset+1 offset+41 offset+41];
        end
        tmp = zeros(dim, Nscale * N);
        idxtmp = 0;
        for i=1:N
            % As 2D image
            imgi = reshape(train_images(:,i), 41, 41);
            for j=1:Nscale
                imgi = imresize(imgi, scales(j));
                imgi = imgi(crops(j,1):crops(j,3), crops(j,2):crops(j,4));
                tmp(:,idxtmp+j) = reshape(imgi, dim, 1);
            end
            idxtmp = idxtmp + Nscale;
        end
        train_images = tmp;
        N = size(train_images, 2);
        
        % Save augmented training data images
        save(trainfile_augmented, 'train_images')
    end
end

% Number of training cases
Ntrain = size(train_images,2);

%% Reduce training set
if Nreduce > 0
    idxtmp = randperm(Ntrain);
    idxtmp = idxtmp(1:Nreduce);
    warning('Reducing training set to %d examples...', length(idxtmp));
    train_images = train_images(:,idxtmp);
    Ntrain = size(train_images,2);
end

%% Create batches
if ~(resume && exist('batches_init', 'var') > 0 && exist('batches', 'var') > 0)
    disp 'Creating batches...'
    batches_init = create_batches(train_images', round(Ntrain/100), 'Method', 'Random');
%     batches = batches_init;
    batches = create_batches(train_images', round(Ntrain/100), 'Method', 'Random');
end

%% Train (or load) network
if resume && exist('data/cifar.mat', 'file')
    disp 'Loading fine tuned network file...'
    load data/cifar.mat;
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
        'LearningRate', learning_rate,...
        'LearningRateMul', learning_rate_mul,...
        'Momentum', momentum,...
        'Regularizer', 0,...
        'Sigma', 0.1,...
        'Width', 41,...
        'Verbose', true,...
        'Visualize', true,...
        'UseGPU', true,...
        'Resume', true);
    save('data/cifar.mat', 'net', 'net_init');
end

wh = sqrt(size(train_images,1)); % Image width/height
if num_hidden(end) < size(train_images,1)
    %% Get a PCA for the training images
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
    idx = randi(Ntrain);
    figure('Name', 'Example')
    subplot(221),imagesc(reshape(train_images(:,idx), [wh wh])),title('Input image')
    subplot(222),imagesc(reshape(pca_train_rec(idx,:)', [wh wh])),title('PCA reconstruction')
    subplot(223),imagesc(reshape(net_train_rec(:,idx)*sigma+mu', [wh wh])),title('NN reconstruction')
    subplot(224),imagesc(reshape(net_fine_train_rec(:,idx)*sigma+mu', [wh wh])),title('Fine-tuned NN reconstruction')
    colormap gray
end
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
idxtmp = randi(Ntrain);
wh = sqrt(size(train_images,1)); % Image width/height
figure('Name', 'Example')
subplot(221),imagesc(reshape(train_images(:,idxtmp), [wh wh])),title('Input image')
subplot(222),imagesc(reshape(pca_train_rec(idxtmp,:)', [wh wh])),title('PCA reconstruction')
subplot(223),imagesc(reshape(net_train_rec(:,idxtmp)*sigma+mu', [wh wh])),title('NN reconstruction')
subplot(224),imagesc(reshape(net_fine_train_rec(:,idxtmp)*sigma+mu', [wh wh])),title('Fine-tuned NN reconstruction')
colormap gray

%% Show some 1-layer unit weights
figure('Name', '1-layer encoder weights before fine tuning')
for i=1:100
    subtightplot(10,10,i),imagesc(reshape(net_init.IW{1}(i,:)',wh,wh))
    axis off
end
colormap gray

figure('Name', '1-layer encoder weights after fine tuning')
for i=1:100
    subtightplot(10,10,i),imagesc(reshape(net.IW{1}(i,:)',wh,wh))
    axis off
end
colormap gray

disp 'All done!'
