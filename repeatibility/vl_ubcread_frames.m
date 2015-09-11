function f = vl_ubcread_frames(file, varargin)
% SIFTREAD Read Lowe's SIFT implementation data files
%   F = VL_UBCREAD(FILE) reads the frames F in the format used by
%   Oxford VGG implementations .
%
%   See also: VL_SIFT(), VL_HELP().

opts.verbosity = 0 ;
opts.format = 'oxford' ;
opts = vl_argparse(opts, varargin) ;

g = fopen(file, 'r');
if g == -1
    error(['Could not open file ''', file, '''.']) ;
end
[header, count] = fscanf(g, '%f', [1 2]) ;
if count ~= 2
    error('Invalid keypoint file header.');
end
switch opts.format
  case 'ubc'
    numKeypoints  = header(1) ;

  case 'oxford'
    numKeypoints  = header(2) ;

  otherwise
    error('Unknown format ''%s''.', opts.format) ;
end

if(opts.verbosity > 0)
	fprintf('%d keypoints.\n', numKeypoints) ;
end

%creates two output matrices
switch opts.format
  case 'ubc'
    P = zeros(4,numKeypoints) ;

  case 'oxford'
    P = zeros(5,numKeypoints) ;
end

%parse tmp.key
for k = 1:numKeypoints

  switch opts.format
    case 'ubc'
      % Record format: i,j,s,th
      [record, count] = fscanf(g, '%f', [1 4]) ;
      if count ~= 4
        error(...
          sprintf('Invalid keypoint file (parsing keypoint %d, frame part)',k) );
      end
      P(:,k) = record(:) ;

    case 'oxford'
      % Record format: x, y, a, b, c such that x' [a b ; b c] x = 1
      [record, count] = fscanf(g, '%f', [1 5]) ;
      if count ~= 5
        error(...
          sprintf('Invalid keypoint file (parsing keypoint %d, frame part)',k) );
      end
      P(:,k) = record(:) ;
  end

end
fclose(g) ;

switch opts.format
  case 'ubc'
    P(1:2,:) = flipud(P(1:2,:)) + 1 ; % i,j -> x,y

    f=[ P(1:2,:) ; P(3,:) ; -P(4,:) ] ;

  case 'oxford'
    P(1:2,:) = P(1:2,:) + 1 ; % matlab origin
    f = P  ;
    f(3:5,:) = inv2x2(f(3:5,:)) ;
end


% --------------------------------------------------------------------
function S = inv2x2(C)
% --------------------------------------------------------------------

den = C(1,:) .* C(3,:) - C(2,:) .* C(2,:) ;
S = [C(3,:) ; -C(2,:) ; C(1,:)] ./ den([1 1 1], :) ;
