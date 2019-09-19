# EHD-convection-2D-openGL
This code solves the 2D electrohydrodynamic convection problem with openGL visualization and control

Windows with Visual Studio:

Project properties => C/C++ Additional Include Directories:
1. Need to add: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2\common\inc for openGL
2. Need to add: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\bin for cufft 

File properties => General => Excluded From Build for kernel.cu, LBM.cu, main_2.cu, poisson.cu
3. Need to compile main.cu, LBM.h, seconds.cpp, seconds.h, interactions.h only
