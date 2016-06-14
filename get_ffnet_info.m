function [W,b,transfer] = get_ffnet_info(ffnet)

W = cell(ffnet.numLayers, 1);
b = cell(ffnet.numLayers, 1);
transfer = cell(ffnet.numLayers, 1);
for i = 1:ffnet.numLayers
    if i == 1
        W{i} = ffnet.IW{1};
        b{i} = ffnet.b{1};
    else
        W{i} = ffnet.LW{i,i-1};
        b{i} = ffnet.b{i};
    end
    transfer{i} = ffnet.layers{i}.transferFcn;
end

end