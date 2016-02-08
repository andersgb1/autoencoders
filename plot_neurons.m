function h = plot_neurons(W, width, rows, varargin)

if nargin < 3, rows = 5; end
assert(round(rows)==rows, 'Non-integer number of rows!')
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Strongest', true, @islogical)
p.parse(varargin{:});
strongest = p.Results.Strongest;

[dim, N] = size(W);
height = dim / width;
assert(round(width) == width, 'Non-integer image width!');
assert(round(height) == height, 'Non-integer image height!');
num_plots = min(rows*rows, N);
strengths = zeros(1,N);
for i=1:N, strengths(i) = norm(W(:,i)); end
if strongest
    [~, idx] = sort(strengths, 'descend');
    figname = sprintf('%i strongest units', num_plots);
else
    [~, idx] = sort(strengths, 'ascend');
    figname = sprintf('%i weakest units', num_plots);
end

% Create fig
h = findobj('type', 'figure', 'name', figname);
if isempty(h), h = figure('Name', figname); end
figure(h)
% h = figure('Name', figname);

% Plot
for i = 1:num_plots
    subtightplot(rows, rows, i)
    w = W(:,idx(i));
    imagesc(reshape(w, [height width]));
    axis equal off
end

% Tweak
colormap gray

end