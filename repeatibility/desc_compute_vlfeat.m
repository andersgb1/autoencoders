clear;clc;

% Image root path
root='repeatibility/graf';

% Image(s) to consider
idxx = {'1', '2', '3', '4', '5', '6'};

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
% Our: custom
detector='custom';
thres = 0.025; % Only applied for VLFeat detectors

% Set custom frame parameters
custom_frames = [];
radii = 20:10:50; % Range of radii
angle_inc = pi/4;
angles = angle_inc:angle_inc:2*pi; % Range of angles

% Descriptor
% sift, liop, patch, none
desc = 'patch';

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
        [frames, descriptors] = vl_covdet(imgs,...
            'EstimateAffineShape', false,...
            'EstimateOrientation', true,...
            'Frames', frames_ox,...
            'Descriptor', desc,...
            'verbose');
        % TODO: Remove duplicates introduced by VLFeat
        % - assuming they are at the bottom of the list
        Nox = size(frames_ox, 2);
        warning('Trying to remove VLFeat''s %d duplicates...', size(frames,2)-Nox);
        frames = frames(:,1:Nox);
        descriptors = descriptors(:,1:Nox);
    elseif strcmpi(detector, 'custom')
        if ~strcmpi(desc, 'patch')
            warning('Custom (many) frames are being computed. It is strongly advised to use the ''patch'' descriptor!\n');
        end
        % Generate frames
        frames_custom = [];
        for rad=radii
            for angle=angles
                xr = rad:rad:size(imgs, 2) - rad + 1;
                yr = rad:rad:size(imgs, 1) - rad + 1;
                [x,y] = meshgrid(xr, yr) ;
                f = [x(:)' ; y(:)' ; rad*ones(1,numel(x)) ; angle*ones(1,numel(x))];
                frames_custom = [frames_custom f];
            end
        end
        % Run
        [frames, descriptors] = vl_covdet(imgs,...
            'EstimateAffineShape', false,...
            'EstimateOrientation', false,...
            'Frames', frames_custom,...
            'Descriptor', desc,...
            'verbose');
    else
        % Run VLFeat detector+descriptor
        [frames, descriptors] = vl_covdet(imgs,...
            'EstimateAffineShape', true,...
            'EstimateOrientation', true,...
            'method', detector,...
            'LaplacianPeakThreshold', thres,...
            'Descriptor', desc,...
            'verbose');
    end

    % Check data
    assert(size(frames,2) == size(descriptors,2));
    N = size(frames,2);

    % Convert all VLFeat frames to Oxford format (removes orientation)
    frames_ox = vl_oell2ox(frames);

    % Save detector outputs
    detfile = [root '/img' idx '.ppm.' detector];
    fprintf('Saving detector outputs to %s...\n', detfile);
    fid = fopen(detfile, 'w');
    fprintf(fid, '1.0\n');
    fprintf(fid, '%i\n', size(frames_ox,2));
    for i=1:N
        fprintf(fid, '%.2f %.2f %f %f %f\n', frames_ox(:,i));
    end
    fclose(fid);

    if ~strcmpi(desc, 'none')
        % Save descriptors
        if strcmpi(desc, 'sift') % SPECIAL CASE: SIFT NAME CONFLICT
            descfile = [root '/img' idx '.ppm.' detector '.sift_vl'];
        else
            descfile = [root '/img' idx '.ppm.' detector '.' desc];
        end
        
        fprintf('Saving %i descriptors to %s...\n', N, descfile);
        fid = fopen(descfile, 'w');
        fprintf(fid, '%i\n', size(descriptors,1));
        fprintf(fid, '%i\n', size(descriptors,2));
        for i=1:N
            fprintf(fid, '%.2f %.2f %f %f %f ', frames_ox(:,i));
            fprintf(fid, '%f ', descriptors(:,i));
            fprintf(fid, '\n');
        end
        fclose(fid);
    end
end

disp 'Done!'
