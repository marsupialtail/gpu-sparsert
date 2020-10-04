# gpu-sparsert

More details will be forthcoming. This will basically work on an Amazon T4 instance with the deep learning AMI.

Dependencies:
- Cnpy. Please install it and update the include path in the scripts.
- Cuda toolchain, etc. 
- Pytorch

To try SpMM, type autotune_float.sh 0
To try sparse convolution, type bash autotune_conv_float.sh 512 512 7 filter_bg4.npy
