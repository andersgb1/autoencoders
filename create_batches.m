function batches = create_batches(X, N, varargin)
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Method', 'Linear', @(x)any(validatestring(x,{'Linear','Random','Cluster','ClusterPCA'})));
p.addParameter('NumClusters', 100, @isnumeric);
p.addParameter('PCAVariation', 95, @isnumeric);
p.addParameter('Resize', 0, @isnumeric);
p.addParameter('Verbose', false, @islogical);
p.parse(varargin{:});
% Get opts
method = p.Results.Method;
num_clusters = p.Results.NumClusters;
pca_variation = p.Results.PCAVariation;
resize = p.Results.Resize;
verbose = p.Results.Verbose;

% Construct batches
if strcmpi(method, 'Linear') || strcmpi(method, 'Random')
    % In the random case, shuffle indices
    if strcmpi(method, 'Linear')
        idx = (1:size(X,1))';
    else
        idx = randperm(size(X,1))';
    end
    % Subdivide indices into the final batches
    sz = round(size(X,1) / N); % Batch size
    batches = cell(1,N);
    ibatch = 1;
    for i = 1:sz:size(X,1)
        idxend = min(i+sz-1, size(X,1));
        batches{ibatch} = idx(i:idxend);
        ibatch = ibatch+1;
    end
elseif strcmpi(method, 'Cluster') || strcmpi(method, 'ClusterPCA')
    % Resize
    if resize > 0
        wh = sqrt(size(X,2));
        assert(round(wh) == wh, 'Resizing only works for square images!');
        img1 = imresize(reshape(X(1,:), wh, wh), resize);
        if verbose, fprintf('Resizing all training cases from %ix%i --> %ix%i...\n', wh, wh, size(img1)); end
        Xresized = zeros(size(X,1), numel(img1));
        Xresized(1,:) = reshape(img1, 1, numel(img1));
        chars = 0;
        for i = 2:size(X,1)
            if verbose
                for j=1:chars, fprintf('\b'); end
                chars = fprintf('%.2f %%',100 * i / size(X,1));
            end
            Xresized(i,:) = reshape(imresize(reshape(X(i,:), wh, wh), resize), 1, numel(img1));
        end
        if verbose, fprintf('\n'); end
        X = Xresized;
    end
    % In the PCA version, we just change X before doing k-means
    if strcmpi(method, 'ClusterPCA')
        % Do PCA
        if verbose, fprintf('Performing PCA...\n'); end
        [coeff,~,~,~,explained,mu] = pca(X);
        % Get the pca_variation % most important components
        assert(pca_variation > 0 && pca_variation <= 100, 'The PCA variation must be a number in ]0,100]!');
        exsum = 0;
        num_components = 1;
        while exsum < pca_variation
            exsum = exsum + explained(num_components);
            num_components = num_components+1;
        end
        num_components = num_components-1;
        if verbose, fprintf('\tUsing %i components!\n', num_components); end
        coeff = coeff(:,1:num_components);
        % Project to subspace
        X = project_pca(X, coeff, mu);
    end
    
    % Do clustering
    if verbose, fprintf('Performing clustering (using %i clusters)...\n', num_clusters); end
    idx = kmedoids(X, num_clusters);
    % Assemble clusters
    clusters = cell(1, num_clusters);
    for i=1:size(X,1), clusters{idx(i)} = [clusters{idx(i)} i]; end
    % If largest cluster is too small, reduce the number of output batches
    cluster_sizes = cellfun(@length, clusters);
    max_cluster_size = max(cluster_sizes);
    if max_cluster_size < N
        warning('Too many clusters (%i) has lead to too small buckets! Only %i batches will be created!', num_clusters, max_cluster_size);
        N = max_cluster_size;
    end
    % Assemble batches
    batches = cell(1, N);
    for i = 1:max_cluster_size
        % Cycle circularly through the batches
        ibatch = mod(i,N);
        if ibatch == 0, ibatch = N; end % At multiples of N, mod returns 0
        for j = 1:num_clusters
            if i <= cluster_sizes(j)
                batches{ibatch} = [batches{ibatch} clusters{j}(i)];
            end
        end
    end
else
    error('Unknown method ''%s''!\n', method);
end
end