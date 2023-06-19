# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import shutil

from distutils.command.build import build as _build
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

PACKAGE_VERSION = os.environ.get("TORCH_MLIR_PYTHON_PACKAGE_VERSION") or "0.0.1"

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):

    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")

class CMakeBuild(build_py):

    def run(self):
        print("run!!!!")
        target_dir = self.build_lib
        cmake_build_dir = os.getenv("TORCH_MLIR_CMAKE_BUILD_DIR")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=False, onerror=None)

        def non_headers(directory, names):
            return [n for n in names if os.path.isfile(os.path.join(directory,n)) \
                    and not n.endswith('.h') \
                    and not n.endswith('.h.inc') \
                    and not n.endswith('.td')]
        
        shutil.copytree(os.path.join(SRC_DIR, 'include'),
                        os.path.join(target_dir, 'torch_mlir', 'include'),
                        symlinks=False,
                        ignore=non_headers)

        shutil.copytree(os.path.join(cmake_build_dir, 'tools', 'torch-mlir-dialects', 'include'),
                        os.path.join(target_dir, 'torch_mlir', 'include'),
                        symlinks=False,
                        ignore=non_headers,
                        dirs_exist_ok=True)

        shutil.copytree(os.path.join(cmake_build_dir, 'tools', 'torch-mlir', 'include'),
                        os.path.join(target_dir, 'torch_mlir', 'include'),
                        symlinks=False,
                        ignore=non_headers,
                        dirs_exist_ok=True)

        def non_torch_libs(directory, names):
            return [n for n in names if "TorchMLIR" not in n]

        shutil.copytree(os.path.join(cmake_build_dir, 'lib'),
                        os.path.join(target_dir, 'torch_mlir', 'lib'),
                        symlinks=False,
                        dirs_exist_ok=True,
                        ignore=non_torch_libs)

class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(build_ext):

    def build_extension(self, ext):
        pass


with open(os.path.join(SRC_DIR, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

print(find_packages(".."))

setup(
    name="torch-mlir-dev",
    version=f"{PACKAGE_VERSION}",
    author="Matthias Gehre",
    author_email="",
    description="First-class interop between PyTorch and MLIR (headers & libraries)",
    long_description=long_description,
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    # For some reason, removing ext_modules here keep all include and lib files
    # from being included in the wheel.
    ext_modules=[
        CMakeExtension("torch_mlir.dev")],
    install_requires=[f"torch-mlir=={PACKAGE_VERSION}"],
    zip_safe=False,
)
