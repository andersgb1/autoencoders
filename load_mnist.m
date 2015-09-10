function [train_images, train_labels, test_images, test_labels] = load_mnist

rootpath = 'mnist';
assert(exist(rootpath, 'dir') == 7);

files= {'train-images-idx3-ubyte',...
    'train-labels-idx1-ubyte',...
    't10k-images-idx3-ubyte',...
    't10k-labels-idx1-ubyte'};

data_cell = cell(1,4);
for i=1:length(files)
    if ~exist([rootpath '/' files{i}], 'file')
        archive = [files{i} '.gz'];
        fprintf('Downloading and unpacking archive %s...\n', archive);
        data = webread(['http://yann.lecun.com/exdb/mnist/' archive]);
        fid = fopen([rootpath '/' archive], 'w');
        fwrite(fid, data);
        gunzip([rootpath '/' archive]);
        fclose(fid);
    end
    
    if mod(i,2) % Odd
        data_cell{i} = loadMNISTImages([rootpath '/' files{i}]);
    else
        data_cell{i} = loadMNISTLabels([rootpath '/' files{i}]);
    end
end

% Write outputs
train_images = data_cell{1};
train_labels = data_cell{2};
test_images = data_cell{3};
test_labels = data_cell{4};
