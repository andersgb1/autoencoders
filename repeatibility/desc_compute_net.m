clear;clc;

path('repeatibility', path);

% Image root path
root='repeatibility';
seqs = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};
    
% Image(s) to consider
idxx = {'1', '2', '4'};

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% detector='haraff';
detector='hesaff';
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
thres = 0.025; % Only applied for VLFeat detectors

%% Load network
% load data/oxford.mat
load data/cifar.mat
enc = get_encoder(net);
enc_init = get_encoder(net_init);

%% Load PCA
% load data/oxford_pca.mat
load data/cifar_pca.mat

%% Loop over image sequences
for seqc = seqs
    % Loops over cells return single cells
    seq = seqc{1};
    
    % Get extension, hacky
    ext = '.ppm';
    if strcmpi(seq, 'boat'), ext = '.pgm'; end
    
    % Loop over images in sequence
    for i = 1:length(idxx)
        % Loops over cells return single cells
        idx = idxx{i};
        
        % Load image
        imgfile = [root '/' seq '/img' idx ext];
        assert(exist(imgfile, 'file') > 0, ['Image file ' imgfile ' does not exist!']);
        img = imread([root '/' seq '/img' idx ext]);
        if size(img,3) == 3, img = rgb2gray(img); end
        imgs = im2single(img);
%         imgs=histeq(imgs);
        
        if strcmpi(detector, 'har') ||...
                strcmpi(detector, 'harlap') ||...
                strcmpi(detector, 'heslap') ||...
                strcmpi(detector, 'haraff') ||...
                strcmpi(detector, 'hesaff')
            % Load detector outputs in the case of Oxford features
            frames_ox = vl_ubcread_frames([root '/' seq '/img' idx ext '.' detector]);
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
        % If network input and patch sizes mismatch, abort
        assert(net.input.size == size(patches,1), 'Inconsistent network and data sizes!');
        
        tic
        
        % Preprocess
        %None
%         patches_norm = patches;
        
        % Using global training mean and std
%             patches_norm = (patches' - repmat(mu, N, 1))';
%             patches_norm = ( (patches' - repmat(mu, N, 1)) / sigma )';
        
        % Using global patch mean and global std
%             patches_norm = patches - repmat(mean(patches, 2), 1, N);
            patches_norm = (patches - repmat(mean(patches, 2), 1, N)) / std(patches(:));
        
        % Using local patch mean and std
%         patches_norm = zeros(size(patches));
%         for i=1:N, patches_norm(:,i) = patches(:,i) - mean(patches(:,i)); end
%         for i=1:N, patches_norm(:,i) = (patches(:,i) - mean(patches(:,i))) / std(patches(:,i)); end

        % Using local patch min-max to normalize each patch to [-1,1]
%         patches_norm = zeros(size(patches));
%         for i=1:N, patches_norm(:,i) = -1 + 2*(patches(:,i) - min(patches(:,i))) / (max(patches(:,i)) - min(patches(:,i))); end
        
        % Encode
        enc_patches = enc(patches_norm);
%         for j=1:N, enc_patches(:,j) = enc_patches(:,j) / sum(abs(enc_patches(:,j))); end
        t = toc;
        fprintf('Encoding time: %f s (%f ms per feature)\n', t, 1000*t/N);
        
        enc_patches_init = enc_init(patches_norm);
        
%         % For showing some reconstructions
%         figure('Name', sprintf('Patches, %s, %s', seq, idx))
%         for j=1:64, subplot(8,8,j),imagesc(reshape(patches_norm(:,j),41,41)); end
%         colormap gray
%         
%         figure('Name', sprintf('Reconstructions, %s, %s', seq, idx))
%         for j=1:64, subplot(8,8,j),imagesc(reshape(net(patches_norm(:,j)),41,41)); end
%         colormap gray
%         
%         drawnow
        
        %% Output
        % Save features
        descfile = [root '/' seq '/img' idx ext '.' detector '.pca'];
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
        
        descfile = [root '/' seq '/img' idx ext '.' detector '.enc'];
        descfile_init = [root '/' seq '/img' idx ext '.' detector '.enc_init'];
        fprintf('Saving %i initial encoder outputs to %s...\n', N, descfile_init);
        fprintf('Saving %i fine tuned encoder outputs to %s...\n', N, descfile);
        fid = fopen(descfile, 'w');
        fid_init = fopen(descfile_init, 'w');
        fprintf(fid, '%i\n', size(enc_patches,1));
        fprintf(fid, '%i\n', size(enc_patches,2));
        fprintf(fid_init, '%i\n', size(enc_patches_init,1));
        fprintf(fid_init, '%i\n', size(enc_patches_init,2));
        for i=1:N
            fprintf(fid, '%.2f %.2f %f %f %f ', frames_ox(:,i));
            fprintf(fid, '%f ', enc_patches(:,i));
            fprintf(fid, '\n');
            fprintf(fid_init, '%.2f %.2f %f %f %f ', frames_ox(:,i));
            fprintf(fid_init, '%f ', enc_patches_init(:,i));
            fprintf(fid_init, '\n');
        end
        fclose(fid);
        fclose(fid_init);
    end
end

disp 'Done!'
