function [f,df] = backprop_loss(target, output, loss)
switch loss
    case 'mse'
        f = mse(target - output);
        if nargout > 1, df = target - output; end
    case 'mae'
        f = mae(target - output);
        if nargout > 1, df = 0.5 * sign(target - output); end
    case {'crossentropy', 'log'}
        f = sum(sum( -target .* log(output + eps) )) / numel(target);
        if nargout > 1, df = -target ./ (output + eps); end
    case {'crossentropy_binary', 'binary_crossentropy'}
        f = sum(sum( -target .* log(output + eps) - (1 - target) .* log(1 - output + eps) )) / numel(target);
        if nargout > 1, df = (target - output) ./ (output .* (1 - output) + eps); end
    otherwise
        error('Unknown loss function: %s!\n', loss);
end