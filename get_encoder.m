function encoder = get_encoder(net)
% Get the encoder part of a network as a Network instance

assert(mod(net.numLayers, 2) == 0, 'Network must have an even number of layers!');
assert(net.inputs{1}.size == net.outputs{end}.size, 'Network must have an equal number of input/output units!');

encoder = get_layer(net, 1);
for i = 2:net.numLayers/2
    encoder = stack(encoder, get_layer(net, i));
end

end