from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='curvature_extension', ext_modules=[CUDAExtension(name='curvature_cuda', sources=['cpp_files/curvature_cuda.cpp', 'cpp_files/curvature_cuda_kernel.cu'], extra_compile_args=['-g'])], cmdclass={'build_ext': BuildExtension})