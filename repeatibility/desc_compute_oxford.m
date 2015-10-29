clear;clc;

path('repeatibility', path);

% Image root path
root='repeatibility';
seqs = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};
    
% Image(s) to consider
idxx = {'1', '2', '4'};

% Detector
% har, harlap, heslap, haraff, hesaff
% detector = 'haraff';
detector='hesaff';
if strcmpi(detector, 'hesaff')
    thresholds={'328', '509', '2078', '1496', '586', '1550', '994', '588'};
elseif strcmpi(detector, 'haraff')
    thresholds={'0', '0', '38000', '17000', '0', '22000', '14000', '2800'};
end

% Descriptor
% jla, sift, gloh, mom, koen, cf, sc, spin, pca, cc, none
descriptors={'sift', 'gloh'};

%% Loop over image sequences
for idxs = 1:length(seqs)
    % Loops over cells return single cells
    seq = seqs{idxs};
    thres = thresholds{idxs};
    
    % Get extension, hacky
    ext = '.ppm';
    if strcmpi(seq, 'boat'), ext = '.pgm'; end
    
    % Loop over images in sequence
    for idx = idxx
        img = [root '/' seq '/img' idx{1} ext];
        assert(exist(img, 'file') > 0, ['Image file ' img ' does not exist!']);
        fprintf('%s...\n', img);
        
        for i = 1:numel(descriptors)
            desc = descriptors{i};
            
            % Output files
            detfile = [img '.' detector];
            descfile = [detfile '.' desc];
            
            % Compute detector outputs
            fprintf('\t%s\n', detfile);
            stat = system(['./repeatibility/detect_points.ln -' detector ' -i ' img ' -o ' detfile ' -thres ' thres ' >/dev/null']);
            assert(stat == 0);
            
            % Compute descriptors
            if ~strcmpi(desc, 'none')
                fprintf('\t\t%s\n', descfile);
                stat = system(['./repeatibility/compute_descriptors.ln -' desc ' -i ' img ' -p1 ' detfile ' -o1 ' descfile ' >/dev/null']);
                assert(stat == 0);
            end
        end
    end
end

disp 'Done!'
