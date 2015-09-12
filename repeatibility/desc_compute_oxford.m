clear;clc;

% Image root path
root='graf';

% Image(s) to consider
idxx = {'1', '2', '3', '4', '5', '6'};

% Detector
% har, harlap, heslap, haraff, hesaff
detector='hesaff';
thres='500';

% Descriptor
% jla, sift, gloh, mom, koen, cf, sc, spin, pca, cc, none
descriptors={'none'};

% Loop over images
for idx = idxx
    img = [root '/img' idx{1} '.ppm'];
    assert(exist(img, 'file') > 0);
    fprintf('%s...\n', img);
    
    for i = 1:numel(descriptors)
        desc = descriptors{i};
        
        % Output files
        detfile = [img '.' detector];
        descfile = [detfile '.' desc];

        % Compute detector outputs
        fprintf('\t%s\n', detfile);
        stat = system(['./detect_points.ln -' detector ' -i ' img ' -o ' detfile ' -thres ' thres ' >/dev/null']);
        assert(stat == 0);

        % Compute descriptors
        if ~strcmpi(desc, 'none')
            fprintf('\t\t%s\n', descfile);
            stat = system(['./compute_descriptors.ln -' desc ' -i ' img ' -p1 ' detfile ' -o1 ' descfile ' >/dev/null']);
            assert(stat == 0);
        end
    end
end

disp 'Done!'
