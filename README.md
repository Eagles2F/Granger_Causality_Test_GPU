Author:Yifan Li
contact:thueeliyifan@gmail.com


Granger Causality Test
===================



Trying to accelerate the process of Granger Causality by GPU.

Files:

%_Raw% : the code for orginal algorithm in C;

%_Cuda% the code for accelerated algorithm in cuda C.

%_Cublas%: the code for accelerated algorithm in Cublas.


Usage:

1. print the command:"export LD_LIBRARY_PATH=/usr/local/MATLAB/R2013a/bin/glnxa64"
2. print the command:"executefile targetfile" for example: ./Gradient_Descent_Raw ./Series.mat
3. you can also try different data in resources.
