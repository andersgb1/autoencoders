clear;clc;close all;

% Image root path
root='graf';

% Image pair to consider
idx1 = '1';
idx2 = '4';

% Detector
% Oxford detectors: har, harlap, heslap, haraff, hesaff
% VLFeat detectors: dog, hessian, hessianlaplace, harrislaplace, multiscalehessian, multiscaleharris
detector = 'hesaff';

% Descriptor(s)
% Oxford descriptors: jla, sift, gloh, mom, koen, cf, sc, spin, pca, cc
% VLFeat descriptors: sift_vl, liop, patch
% Our descriptors: pca
descs = {'sift', 'liop', 'pca', 'enc'};

% Matching type
% nn, sim, ratio
mtype = 'sim'; % Mikolajczyk et al. used 'sim', while the LIOP folks used 'ratio'

% Allocate plot data
precision_nn = zeros(20, numel(descs));
recall_nn = zeros(20, numel(descs));
precision_sim = zeros(20, numel(descs));
recall_sim = zeros(20, numel(descs));
precision_ratio = zeros(20, numel(descs));
recall_ratio = zeros(20, numel(descs));

% Start gathering descriptor matching results
for i = 1:numel(descs)
    desc = descs{i};
    
    % Derived file names
    img1 = [root '/img' idx1 '.ppm'];
    img2 = [root '/img' idx2 '.ppm'];
    H = [root '/H' idx1 'to' idx2 'p'];
    det1 = [img1 '.' detector];
    det2 = [img2 '.' detector];
    desc1 = [det1 '.' desc];
    desc2 = [det2 '.' desc];
    
    assert(exist(img1, 'file') > 0);
    assert(exist(img2, 'file') > 0);
    assert(exist(det1, 'file') > 0);
    assert(exist(det2, 'file') > 0);
    assert(exist(desc1, 'file') > 0);
    assert(exist(desc2, 'file') > 0);

    [v_overlap,v_repeatability,v_nb_of_corespondences,matching_score,nb_of_matches,twi] = repeatability(...
        desc1,...
        desc2,...
        H,...
        img1,...
        img2,...
        0);

    [correct_match_nn,total_match_nn,correct_match_sim,total_match_sim,correct_match_rn,total_match_rn] = descperf(...
        desc1,...
        desc2,...
        H,...
        img1,...
        img2,...
        v_nb_of_corespondences(5),...
        twi);

    % Get precision recall
    if strcmpi(mtype, 'nn')
        corresp = v_nb_of_corespondences(5);
        recall_nn(:,i) = correct_match_nn ./ corresp;
        precision_nn(:,i) = (total_match_nn - correct_match_nn) ./ total_match_nn;
    elseif strcmpi(mtype, 'sim')
        corresp = sum(sum(twi));
        recall_sim(:,i) = correct_match_sim ./ corresp;
        precision_sim(:,i) = (total_match_sim - correct_match_sim) ./ total_match_sim;
    elseif  strcmpi(mtype, 'ratio')
        corresp = v_nb_of_corespondences(5);
        % This has potentially fewer samples than 20
        Nratio = size(correct_match_rn, 2);
        if Nratio == 20
            recall_ratio(:,i) = correct_match_rn ./ corresp;
            precision_ratio(:,i) = (total_match_rn - correct_match_rn) ./ total_match_rn;
        elseif Nratio < 20
            recall_ratio(1:Nratio,i) = correct_match_rn ./ corresp;
            precision_ratio(1:Nratio,i) = (total_match_rn - correct_match_rn) ./ total_match_rn;
            recall_ratio(Nratio+1:end,:) = nan;
            precision_ratio(Nratio+1:end,:) = nan;
        else
            error('Invalid number of ratio PR samples!');
        end
    else
        error('No such matching strategy: %s!', mtype);
    end
end

% Plot
if strcmpi(mtype, 'nn')
    plot(precision_nn, recall_nn, 'LineWidth', 1.5);
elseif strcmpi(mtype, 'sim')
    plot(precision_sim, recall_sim, 'LineWidth', 1.5);
elseif strcmpi(mtype, 'ratio')
    plot(precision_ratio, recall_ratio, 'LineWidth', 1.5);
else
    error('No such matching strategy: %s!', mtype);
end
legend(descs, 'Interpreter', 'none')
title([root ' ' idx1 '-' idx2 ' (' mtype ', ' num2str(corresp) ' correspondences)'])
xlabel('1-Precision');ylabel('Recall');
xlim([0 1]);ylim([0 1]);
ax=gca;ax.XTick=0:0.1:1;ax.YTick=0:0.1:1;
grid;

disp 'Done!'
