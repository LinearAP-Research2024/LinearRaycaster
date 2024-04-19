from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='linearcaster_cpp',
      ext_modules=[cpp_extension.CppExtension('linearcaster_cpp', ['linearcaster.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
