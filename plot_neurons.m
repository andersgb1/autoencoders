function h = plot_neurons(W, width, rows, varargin)

if nargin < 3, rows = 5; end
assert(round(rows)==rows, 'Non-integer number of rows!')
p = inputParser;
p.CaseSensitive = false;
p.addParameter('Strongest', true, @islogical)
p.addParameter('Strengths', [], @isfloat)
p.addParameter('Name', '', @ischar)
p.parse(varargin{:});
strongest = p.Results.Strongest;
strengths = p.Results.Strengths;
name = p.Results.Name;

[dim, N] = size(W);
height = dim / width;
assert(round(width) == width, 'Non-integer image width!');
assert(round(height) == height, 'Non-integer image height!');
num_plots = min(rows*rows, N);
if isempty(strengths)
    strengths = zeros(1,N);
    for i=1:N, strengths(i) = norm(W(:,i)); end
else
    assert(numel(strengths)==N);
end
if strongest
    [~, idx] = sort(strengths, 'descend');
    if isempty(name), name = sprintf('%i strongest units', num_plots); end
else
    [~, idx] = sort(strengths, 'ascend');
    if isempty(name), name = sprintf('%i weakest units', num_plots); end
end

if isempty(name), name = sprintf('%i units', num_plots); end

% Create fig
h = findobj('type', 'figure', 'name', name);
if isempty(h)
    h = figure('Name', name);
else
    set(0, 'CurrentFigure', h);
end

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