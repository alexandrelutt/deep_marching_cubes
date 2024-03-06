from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='smoothness_extension', ext_modules=[CUDAExtension(name='smoothness_cuda', sources=['cpp_files/smoothness_cuda.cpp', 'cpp_files/smoothness_cuda_kernel.cu'], extra_compile_args=['-g'])], cmdclass={'build_ext': BuildExtension})