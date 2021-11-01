%MFEAT_MOR 2000 objects with 6 features in 10 classes
%
%	A = MFEAT_MOR
%
% Load the dataset in A. These are some morphological features.
%
% See also DATASETS, PRDATASETS, MFEAT

% Copyright: R.P.W. Duin, r.p.w.duin@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function a = mfeat_mor

prdatasets(mfilename,1);
a = pr_dataset('mfeat_mor');
a = setname(a,'MFEAT Morphological Features');
