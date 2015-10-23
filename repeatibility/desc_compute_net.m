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

%% Load network
load data/oxford.mat
enc = get_encoder(net);

%% Load PCA
load data/oxford_pca.mat

%% Loop over images
for idxc = idxx
    % Loops over cells return single cells
    idx = idxc{1};

    % Load image
    imgfile = [root '/img' idx '.ppm'];
    assert(exist(imgfile, 'file') > 0, ['Image file ' imgfile ' does not exist!']);
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
    % Project and transpose back to column-major (one feature per column)
    pca_patches = project_pca(patches', c_pca, mu_pca)';
    
    %% NN section    
    % If network input and patch sizes mismatch, rescale all patches
    if net.input.size ~= size(patches,1)
        if idx == '1'
            warning('Inconsistent network and data sizes! Rescaling all patches...');
        end
        pwh = sqrt(size(patches,1)); % Patch width/height
        assert(mod(net.input.size,2) == 0); % Network input size must be even
        nwh = sqrt(net.input.size); % Network width/height
        tmp = zeros(net.input.size, N);
        for i=1:N
            patchi = reshape(patches(:,i), pwh, pwh);
            tmp(:,i) = reshape(imresize(patchi, [nwh nwh]), net.input.size, 1);
        end
        patches = tmp;
    end
    
    % Encode
    patches = ( (patches' - repmat(mu, N, 1)) / sigma )';
    enc_patches = enc(patches);
    
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
