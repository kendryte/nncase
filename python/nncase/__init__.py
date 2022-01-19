# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""nncase."""
import ctypes
import sys
import os


def _add_dllpath():
    if sys.platform == 'win32':
        cur_dir = os.path.dirname(__file__)
        site_pkgs_dir = os.path.join(cur_dir, '..')
        dll_paths = [site_pkgs_dir]
        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(0x0001)
        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p
        for dll_path in dll_paths:
            if sys.version_info >= (3, 8):
                os.add_dll_directory(dll_path)
            elif with_load_library_flags:
                res = kernel32.AddDllDirectory(dll_path)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += f' Error adding "{dll_path}" to the DLL directories.'
                    raise err


_add_dllpath()

from . import base
from _nncase import test_target
from _nncase import Compiler, CompileOptions, ImportOptions, PTQTensorOptions, MemoryRange, RuntimeTensor, GraphEvaluator, DumpRangeTensorOptions
from .simulator import Simulator
