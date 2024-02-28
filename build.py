from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='extension', ext_modules=[CUDAExtension(name='cuda_extension', sources=['cpp_files/occtopology_cuda.cpp', 'cpp_files/occtopology_cuda_kernel.cu'], extra_compile_args=['-g'])], cmdclass={'build_ext': BuildExtension})