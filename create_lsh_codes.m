% dataset_name should be one of: sift_1M / sift_1B / gist_1M / gist_80M.
% You should download the datasets separately.
dataset_name = 'sift_1B';

% nb controls the lenght of the binary codes generated.
% nb can be set from outside.
if (~exist('nb', 'var'))
  nb = 64;
end

% Where the corresponding datasets are stored:
TINY_HOME = 'data/tiny'; 		% the root of 80 million tiny images dataset
INRIA_HOME = 'data/inria';		% the root of INIRA BIGANN datasets

% Where the output matrix of binary codes should be stored:
outputdir = 'codes/lsh';

% CACHE_DIR is used to store the data mean for the datasets.
CACHE_DIR = 'cache';

if (~exist(outputdir, 'file'))
  mkdir(outputdir);
end
if (~exist(CACHE_DIR, 'file'))
  mkdir(CACHE_DIR);
end

addpath matlab;
if (strcmp(dataset_name, 'sift_1B') || strcmp(dataset_name, 'sift_1M') || strcmp(dataset_name, 'gist_1M'))
  addpath([INRIA_HOME, '/matlab']);
else
  addpath([TINY_HOME, '/code']);
end

if strcmp(dataset_name, 'sift_1M')
  dataset = 'ANN_SIFT1M';
  datahome = INRIA_HOME;
  N = 10^6;
elseif strcmp(dataset_name, 'sift_1B')
  dataset = 'ANN_SIFT1B';
  datahome = INRIA_HOME;
  N = 10^9;
elseif strcmp(dataset_name, 'sift_1B_tr')
  dataset = 'ANN_SIFT1B';
  datahome = INRIA_HOME;
  N = 10^8;
elseif strcmp(dataset_name, 'gist_1M')
  dataset = 'ANN_GIST1M';
  datahome = INRIA_HOME;
  N = 10^6;
elseif strcmp(dataset_name, 'gist_80M')
  dataset = '80M';
  datahome = TINY_HOME;
  N = 79*10^6;
end

if ~exist([CACHE_DIR, '/', dataset_name, '_mean.mat'], 'file')
  fprintf('Computing the data mean for the %s dataset... \n', dataset_name);
  if strcmp(dataset_name, 'sift_1M')
    trdata = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_learn.fvecs']);
    learn_mean = mean(trdata, 2);
    save([CACHE_DIR, '/sift_1M_mean'], 'learn_mean');
  elseif strcmp(dataset_name, 'sift_1B')
    Ntraining = 10^8;
    nbuffer = 10^6;
    for i=1:floor(Ntraining/nbuffer)
      fprintf('%d/%d\r', i, floor(Ntraining/nbuffer));
      trdatai = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
      learn_meani(:,i) = sum(trdatai, 2, 'double');
    end
    learn_mean = sum(learn_meani, 2, 'double');
    learn_mean = learn_mean / Ntraining;
    clear trdatai learn_meani;
    save([CACHE_DIR, '/sift_1B_mean'], 'learn_mean');
  elseif strcmp(dataset_name, 'gist_1M')
    trdata = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_learn.fvecs']);
    learn_mean = mean(trdata, 2);
    save([CACHE_DIR, '/gist_1M_mean'], 'learn_mean');
  elseif strcmp(dataset_name, 'gist_80M')
    trdata = read_tiny_gist_binary(1:10^7);
    learn_mean = mean(trdata, 2);
    clear trdata;
    save([CACHE_DIR, '/gist_80M_mean'], 'learn_mean');
    
    perm = randperm(79302017);
    perm(1:79*10^6) = sort(perm(1:79*10^6));
    save([CACHE_DIR, '/gist_80M_mean'], 'perm', '-append');
  else
    fprintf('dataset not supported.\n');
    exit;
  end
  fprintf('done.     \n');
else
  load([CACHE_DIR, '/', dataset_name, '_mean']);
end

nd = size(learn_mean, 1);
W = [randn(nb, nd) zeros(nb, 1)];	% Random projection-based hashing (LSH) preserves angles.
					% One can load W from outside too
nbuffer = 10^6;
B = zeros(ceil(nb/8), N, 'uint8');

fprintf('Computing %d-bit binary codes...\n', nb);
for i=1:floor(N/nbuffer)
  fprintf('%d/%d\r', i, floor(N/nbuffer));
  if strcmp(dataset_name, 'sift_1M')
    base = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]); 
  elseif strcmp(dataset_name, 'sift_1B')
    base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_base.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
  elseif strcmp(dataset_name, 'sift_1B_tr')
    base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
  elseif strcmp(dataset_name, 'gist_1M')
    base = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]); 
  elseif strcmp(dataset_name, 'gist_80M')
    base = read_tiny_gist_binary( perm(((i-1)*nbuffer+1):((i)*nbuffer)) );
  end
  base = double(base);  
  base = bsxfun(@minus, base, learn_mean);
  
  B1 = (W * [base; ones(1, size(base,2))]) > 0;
  B1 = compactbit(B1);
  
  B(:, (i-1)*nbuffer+1:(i)*nbuffer) = B1;
end

query = [];
if strcmp(dataset_name, 'sift_1M')
  query = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_query.fvecs']);
elseif strcmp(dataset_name, 'sift_1B')
  query = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_query.bvecs']);
elseif strcmp(dataset_name, 'gist_1M')
  query = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_query.fvecs']);
  elseif strcmp(dataset_name, 'gist_80M')
    query = read_tiny_gist_binary( perm([(79302017-10000+1):79302017]) );
end
if (isempty(query))
  Q = [];
else
  query = bsxfun(@minus, query, learn_mean);
  Q = (W * [query; ones(1, size(query,2))] > 0);
  Q = compactbit(Q);
end

fprintf('storing the codes in the file %s ...', [outputdir, '/lsh_', num2str(nb), '_', dataset_name]);
save([outputdir, '/lsh_', num2str(nb), '_', dataset_name], 'B', 'Q', 'W', 'learn_mean', '-v7.3');
clear B Q W;
fprintf('done.\n');

% Read a set of vectors stored in the fvec format (int + n * float)
% The function returns a set of output vector (one vector per column)
%
% Syntax: 
%   v = fvecs_read (filename)     -> read all vectors
%   v = fvecs_read (filename, n)  -> read n vectors 
%   v = fvecs_read (filename, [a b]) -> read the vectors from a to b (indices starts from 1)
function v = fvecs_read (filename, bounds)

% open the file and count the number of descriptors
fid = fopen (filename, 'rb');
 
if fid == -1
  error ('I/O error : Unable to open the file %s\n', filename)
end

% Read the vector size
d = fread (fid, 1, 'int');
vecsizeof = 1 * 4 + d * 4;

% Get the number of vectrors
fseek (fid, 0, 1);
a = 1;
bmax = ftell (fid) / vecsizeof;
b = bmax;

if nargin >= 2
  if length (bounds) == 1
    b = bounds;
   
  elseif length (bounds) == 2
    a = bounds(1);
    b = bounds(2);    
  end
end

assert (a >= 1);
if b > bmax
  b = bmax;
end

if b == 0 | b < a
  v = [];
  fclose (fid);
  return;
end

% compute the number of vectors that are really read and go in starting positions
n = b - a + 1;
fseek (fid, (a - 1) * vecsizeof, -1);

% read n vectors
v = fread (fid, (d + 1) * n, 'float=>single');
v = reshape (v, d + 1, n);

% Check if the first column (dimension of the vectors) is correct
assert (sum (v (1, 2:end) == v(1, 1)) == n - 1);
v = v (2:end, :);
  
fclose (fid);

end


% Read a set of vectors stored in the ivec format (int + n * int)
% The function returns a set of output vector (one vector per column)
%
% Syntax: 
%   v = ivecs_read (filename)     -> read all vectors
%   v = ivecs_read (filename, n)  -> read n vectors 
%   v = ivecs_read (filename, [a b]) -> read the vectors from a to b (indices starts from 1)
function v = ivecs_read (filename, bounds)

% open the file and count the number of descriptors
fid = fopen (filename, 'rb');
 
if fid == -1
  error ('I/O error : Unable to open the file %s\n', filename)
end

% Read the vector size
d = fread (fid, 1, 'int');
vecsizeof = 1 * 4 + d * 4;

% Get the number of vectrors
fseek (fid, 0, 1);
a = 1;
bmax = ftell (fid) / vecsizeof;
b = bmax;

if nargin >= 2
  if length (bounds) == 1
    b = bounds;
   
  elseif length (bounds) == 2
    a = bounds(1);
    b = bounds(2);    
  end
end

assert (a >= 1);
if b > bmax
  b = bmax;
end

if b == 0 | b < a
  v = [];
  fclose (fid);
  return;
end

% compute the number of vectors that are really read and go in starting positions
n = b - a + 1;
fseek (fid, (a - 1) * vecsizeof, -1);

% read n vectors
v = fread (fid, (d + 1) * n, 'int=>double');
v = reshape (v, d + 1, n);

% Check if the first column (dimension of the vectors) is correct
assert (sum (v (1, 2:end) == v(1, 1)) == n - 1);
v = v (2:end, :);
  
fclose (fid);

end