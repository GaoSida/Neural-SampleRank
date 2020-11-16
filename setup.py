from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cpp_sampling',
      ext_modules=[cpp_extension.CppExtension(
            name='cpp_sampling',
            sources=['cpp/gibbs_sampling.cpp',
                     'cpp/pybind.cpp',
                     'cpp/sample_rank.cpp'],
            include_dirs=['cpp/']),
                   cpp_extension.CppExtension(
            name='cpp_tests',
            sources=['cpp_tests/test_factor_graph.cpp',
                     'cpp_tests/pybind.cpp',
                     'cpp_tests/test_gibbs_sampling.cpp',
                     'cpp_tests/test_margin_metric.cpp',
                     'cpp/gibbs_sampling.cpp',
                     'cpp/sample_rank.cpp'],
            include_dirs=['cpp/'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
