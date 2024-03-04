from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='distance_extension', ext_modules=[CUDAExtension(name='distance_cuda', sources=['cpp_files/distance_cuda.cpp', 'cpp_files/distance_cuda_kernel.cu'], extra_compile_args=['-g'])], cmdclass={'build_ext': BuildExtension})