from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='utils_extension', ext_modules=[CUDAExtension(name='cpp_utils', sources=['cpp_files/cpp_utils.cpp'], extra_compile_args=['-g'])], cmdclass={'build_ext': BuildExtension})