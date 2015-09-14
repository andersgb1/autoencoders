clear;clc;

% Image root path
root='repeatibility/graf';

% Image(s) to consider
idxx = {'1', '2', '3', '4', '5', '6'};

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
detector='hesaff';
thres = 0.025; % Only applied for VLFeat detectors

% Loop over images
for idxc = idxx
    % Loops over cells return single cells
    idx = idxc{1};

    % Load image
    assert(exist([root '/img' idx '.ppm'], 'file') > 0);
    img = imread([root '/img' idx '.ppm']);
    imgs = im2single(rgb2gray(img));

    if strcmpi(detector, 'har') ||...
            strcmpi(detector, 'harlap') ||...
            strcmpi(detector, 'heslap') ||...
            strcmpi(detector, 'haraff') ||...
            strcmpi(detector, 'hesaff')
        % Load detector outputs in the case of Oxford features
        frames_ox = vl_ubcread_frames([root '/img' idx '.ppm.' detector]);
        % Run
        [frames, patches] = vl_covdet(imgs,...
            'EstimateAffineShape', false,...
            'EstimateOrientation', true,...
            'Frames', frames_ox,...
            'Descriptor', 'patch',...
            'verbose');
        % TODO: Remove duplicates introduced by VLFeat
        % - assuming they are at the bottom of the list
        Nox = size(frames_ox, 2);
        warning('Trying to remove VLFeat''s %d duplicates...', size(frames,2)-Nox);
        frames = frames(:,1:Nox);
        patches = patches(:,1:Nox);
    else
        % Run VLFeat detector+descriptor
        [frames, patches] = vl_covdet(imgs,...
            'EstimateAffineShape', true,...
            'EstimateOrientation', true,...
            'method', detector,...
            'LaplacianPeakThreshold', thres,...
            'Descriptor', 'patch',...
            'verbose');
    end

    % Check data
    assert(size(frames,2) == size(patches,2));
    N = size(frames,2);
    
    %% PCA section
    % Get a PCA from first image
    if idx == '1'
        disp 'Computing PCA space from first image...';
        [c,~,~,~,~,mu] = pca(patches', 'NumComponents', 30);
    end
    assert(exist('c', 'var') > 0 && exist('mu', 'var') > 0, 'PCA from first image non-existent!');
    % Project
    pca_patches = (patches'-repmat(mu,N,1)) * c;
    pca_patches = pca_patches'; % Change to column-major (one feature per column)
    
    %% NN section
    if idx == '1'
        % Get network
        load data/patch_net_fine.mat
        
        % Get encoder
        enc_fine = stack(get_layer(net_fine, 1), get_layer(net_fine,2), get_layer(net_fine, 3), get_layer(net_fine, 4));
    end
    assert(exist('net_fine', 'var') > 0 && exist('enc_fine', 'var') > 0);
    
    % If network input and patch sizes mismatch, rescale all patches
    if net_fine.input.size ~= size(patches,1)
        if idx == '1'
            warning('Inconsistent network and data sizes! Rescaling all patches...');
        end
        pwh = sqrt(size(patches,1)); % Patch width/height
        assert(mod(net_fine.input.size,2) == 0); % Network input size must be even
        nwh = sqrt(net_fine.input.size); % Network width/height
        tmp = zeros(net_fine.input.size, N);
        for i=1:N
            patchi = reshape(patches(:,i), pwh, pwh);
            tmp(:,i) = reshape(imresize(patchi, [nwh nwh]), net_fine.input.size, 1);
        end
        patches = tmp;
    end
    
    % Encode
    enc_patches = enc_fine(patches);
    
    %% Output
    % Save features
    descfile = [root '/img' idx '.ppm.' detector '.pca'];
    fprintf('Saving %i PCA patches to %s...\n', N, descfile);
    frames_ox = vl_oell2ox(frames);
    
    fid = fopen(descfile, 'w');
    fprintf(fid, '%i\n', size(pca_patches,1));
    fprintf(fid, '%i\n', size(pca_patches,2));
    for i=1:N
        fprintf(fid, '%.2f %.2f %f %f %f ', frames_ox(:,i));
        fprintf(fid, '%f ', pca_patches(:,i));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    descfile = [root '/img' idx '.ppm.' detector '.enc'];
    fprintf('Saving %i encoder output to %s...\n', N, descfile);
    fid = fopen(descfile, 'w');
    fprintf(fid, '%i\n', size(enc_patches,1));
    fprintf(fid, '%i\n', size(enc_patches,2));
    for i=1:N
        fprintf(fid, '%.2f %.2f %f %f %f ', frames_ox(:,i));
        fprintf(fid, '%f ', enc_patches(:,i));
        fprintf(fid, '\n');
    end
    fclose(fid);
end

disp 'Done!'
