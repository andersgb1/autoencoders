function frames = vl_oell2ox(oframes)

N = size(oframes,2);
frames = zeros(5, N);

% Remove the orientation
for i=1:N
    A = reshape(oframes(3:6,i), [2 2]);
    theta = -atan2(A(1,1), A(1,2)) - pi/2;
    A = A * [cos(theta) -sin(theta) ; sin(theta) cos(theta)];
    oframes(3:6,i) = reshape(A, [4 1]);
end

% Go from VLFeat's weird affine matrix format to ellipse
frames(1:2,:) = oframes(1:2,:);
frames(3,:) = oframes(3,:).^2;
frames(4,:) = oframes(3,:) .* oframes(4,:);
frames(5,:) = oframes(6,:).^2 + oframes(4,:).*oframes(4,:);

% Use Oxford's 0-indexing and the weird inverted ellipse parameters
for i=1:N
    frames(1,i) = frames(1,i) - 1;
    frames(2,i) = frames(2,i) - 1;
    A = inv([frames(3,i) frames(4,i);frames(4,i) frames(5,i)]);
    frames(3,i) = A(1,1);
    frames(4,i) = A(1,2);
    frames(5,i) = A(2,2);
end
