# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TORCH_MLIR_PYTHON"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
if "TEST_SRC_PATH" in os.environ:
    config.environment["TEST_SRC_PATH"] = os.environ["TEST_SRC_PATH"]

# path to our python operation library
config.environment["TEST_BUILD_PATH"] = os.path.join(config.torch_mlir_obj_root)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.torch_mlir_obj_root, "test")

# Python configuration with sanitizer requires some magic preloading. This will only work on clang/linux.
# TODO: detect Darwin/Windows situation (or mark these tests as unsupported on these platforms).
if os.environ.get("IS_ASAN") and "Linux" in config.host_os:
    # Write shim script that preloads the necessary shared object for ASAN tests. Fallback to such script for two reasons:
    #
    # (1) Provide full support for LLVM's test utils like `not`, which are prepended to the original statement containing the `LD_PRELOAD` env definition.
    #     Having environment definitions in the middle of a command line is syntactically illegal.
    # (2) Mitigate issues with LIT's internal shell that puts single quotes around the environment definition,
    #     which leads to malformed command lines:
    #     `LD_PRELOAD=$(/usr/bin/clang++-17' '-print-file-name=libclang_rt.asan-x86_64.so)' python (...)`
    with open("python-asan-shim", "w") as file:
        file.write(
            f"#!/usr/bin/env bash\nLD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {config.python_executable} $@\n"
        )
    os.chmod(os.path.abspath("python-asan-shim"), 0o700)
    config.python_executable = os.path.abspath("python-asan-shim")
# On Windows the path to python could contains spaces in which case it needs to
# be provided in quotes.  This is the equivalent of how %python is setup in
# llvm/utils/lit/lit/llvm/config.py.
elif "Windows" in config.host_os:
    config.python_executable = '"%s"' % (config.python_executable)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%PYTHON", config.python_executable))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "lit.cfg.py",
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
]

if not bool(int(os.environ.get("TORCH_MLIR_ENABLE_LTC", 0))):
    config.excludes.append("lazy_backend")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.torch_mlir_obj_root, "test")
config.torch_mlir_tools_dir = os.path.join(config.torch_mlir_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.torch_mlir_python_packages_dir, "torch_mlir"),
    ],
    append_path=True,
)


tool_dirs = [config.torch_mlir_tools_dir, config.llvm_tools_dir]
tools = [
    "torch-mlir-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
