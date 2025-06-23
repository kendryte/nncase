#!/usr/bin/env python3
"""
Base classes and utilities for generating NTT test cases.
"""

import os
from collections import namedtuple

# is_contiguous: bool
# non_contiguous_dim: int or None
# big_tensor_op: str or None - How to build the big tensor at given non_contiguous_dim
Continuity = namedtuple('Continuity', ['is_contiguous', 'non_contiguous_dim', 'big_tensor_op'])
DataType = namedtuple('DataType', ['cpp_type', 'name_suffix', 'min_val', 'max_val'])

ALL_DATATYPES = [
    DataType('bool', 'Bool', 'false', 'true'),
    DataType('uint8_t', 'Uint8', '0', '255'),
    DataType('uint16_t', 'Uint16', '0', '65535'),
    DataType('uint32_t', 'Uint32', '0', '100000'),
    DataType('uint64_t', 'Uint64', '0', '1000000'),
    DataType('int8_t', 'Int8', '-127', '127'),
    DataType('int16_t', 'Int16', '-32767', '32767'),
    DataType('int32_t', 'Int32', '-100000', '100000'),
    DataType('int64_t', 'Int64', '-1000000', '1000000'),
    DataType('half', 'Float16', '-65504.0', '65504.0'),
    DataType('float', 'Float32', '-3.4e38', '3.4e38'),
    DataType('double', 'Float64', '-1.7e308', '1.7e308'),
    DataType('bfloat16', 'Bfloat16', '-3.3e38_bf16', '3.3e38_bf16'),
    DataType('float_e4m3_t', 'Float8e4m3', 'float_e4m3_t(-448.0f)', 'float_e4m3_t(448.0f)'),
    DataType('float_e5m2_t', 'Float8e5m2', 'float_e5m2_t(-57344.0f)', 'float_e5m2_t(57344.0f)'),
]

class BaseTestGenerator:
    def __init__(self):
        self.test_cases = []

    def generate_shape_init(self, shape_type, dims):
        if shape_type == "fixed":
            dim_strs = [f"{d}" for d in dims]
            return f"ntt::fixed_shape_v<{', '.join(dim_strs)}>"
        else:  # dynamic
            dim_strs = [str(d) for d in dims]
            return f"ntt::make_shape({', '.join(dim_strs)})"

    def generate_tensor_init(self, datatype, shape_type, dims, continuity, var_name, vector_rank, P=None, axes_count=1):
        code = []
        shape_expr = self.generate_shape_init(shape_type, dims)

        # Determine element type based on vector_rank
        if vector_rank == 0:
            element_cpp_type = datatype.cpp_type
        elif vector_rank == 1:
            if P is None:
                raise ValueError("P must be provided for vector_rank 1")
            element_cpp_type = f"ntt::vector<{datatype.cpp_type}, {P}>"
        elif vector_rank > 1:
            if P is None or axes_count is None:
                raise ValueError("P and axes_count must be provided for vector_rank > 1")
            ps = ', '.join([str(P)] * axes_count)
            element_cpp_type = f"ntt::vector<{datatype.cpp_type}, {ps}>"
        else:
            raise ValueError(f"Invalid vector_rank: {vector_rank}")

        if continuity.is_contiguous:
            code.append(f"alignas(32) auto {var_name} = ntt::make_tensor<{element_cpp_type}>({shape_expr});")
            code.append(f"NttTest::init_tensor({var_name}, min_input, max_input);")
        else:  # non-contiguous
            big_dims = dims.copy()
            dim_to_change = continuity.non_contiguous_dim
            op = continuity.big_tensor_op

            if dim_to_change is not None and op is not None and dim_to_change < len(big_dims):
                big_dims[dim_to_change] = f"({big_dims[dim_to_change]}) {op}"

            big_shape_expr = self.generate_shape_init(shape_type, big_dims)

            code.append(f"// Create non-contiguous tensor (on dimension {dim_to_change})")
            code.append(f"alignas(32) auto big_tensor = ntt::make_tensor<{element_cpp_type}>({big_shape_expr});")
            code.append(f"NttTest::init_tensor(big_tensor, min_input, max_input);")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view_from_address<{element_cpp_type}>(")
            code.append(f"    big_tensor.elements().data(),")
            code.append(f"    {shape_expr},")
            code.append(f"    big_tensor.strides());")

        return code

    def generate_header(self):
        return '''/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

'''

    def generate_footer(self):
        return '''int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
'''

def generate_cmake_list(directory, filenames):
    """generate a .cmake file that contains the list of generated test files"""
    cmake_list_path = os.path.join(directory, "generated_tests.cmake")
    with open(cmake_list_path, "w") as f:
        f.write("# This file is generated automatically. DO NOT EDIT.\n")
        f.write("set(GENERATED_TEST_SOURCES\n")
        for name in filenames:
            f.write(f"    ${{CMAKE_CURRENT_LIST_DIR}}/{name}\n") # use relative path to current CMakeLists.txt
        f.write(")\n")
    print(f"Generated CMake list: {cmake_list_path}")
