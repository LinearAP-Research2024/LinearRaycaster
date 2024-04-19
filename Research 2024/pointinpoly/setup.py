from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pointinpoly_cpp',
      ext_modules=[cpp_extension.CppExtension('pointinpoly_cpp', ['pointinpoly.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
