function h = plot_neurons(W, width)
[dim, N] = size(W);
height = dim / width;
assert(round(width) == width, 'Non-integer image width!');
assert(round(height) == height, 'Non-integer image height!');
num_plots = min(25, N);
strengths = zeros(1,N);
for i=1:N, strengths(i) = norm(W(:,i)); end
[~, idx] = sort(strengths, 'descend');

% Create fig
figname = sprintf('%i strongest units', num_plots);
h = findobj('type', 'figure', 'name', figname);
if isempty(h), h = figure('Name', figname); end
figure(h)

% Plot
for i = 1:num_plots
    subtightplot(5, 5, i)
    w = W(:,idx(i));
    imagesc(reshape(w, [height width]));
    axis equal off
end

% Tweak
colormap gray

end