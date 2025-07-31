#!/usr/bin/env python3
"""
Base classes and utilities for generating NTT test cases.
"""

import os
from collections import namedtuple
from typing import List, Optional

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
    DataType('half', 'Float16', 'half(-65504.0f)', 'half(65504.0f)'),
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
        elif vector_rank > 0:
            if P is None:
                raise ValueError("P must be provided for vector_rank > 0")
            
            # The rank of the vector is determined by vector_rank.
            ps = ', '.join([f"P"] * vector_rank)
            element_cpp_type = f"ntt::vector<{datatype.cpp_type}, {ps}>"
        else:
            raise ValueError(f"Invalid vector_rank: {vector_rank}")

        if continuity.is_contiguous:
            code.append(f"auto {var_name} = ntt::make_tensor<{element_cpp_type}>({shape_expr});")
            code.append(f"NttTest::init_tensor({var_name}, min_input, max_input);")
        else:  # non-contiguous
            big_dims = dims.copy()
            dim_to_change = continuity.non_contiguous_dim
            op = continuity.big_tensor_op

            if dim_to_change is not None and op is not None and dim_to_change < len(big_dims):
                big_dims[dim_to_change] = f"({big_dims[dim_to_change]}) {op}"

            big_shape_expr = self.generate_shape_init(shape_type, big_dims)

            code.append(f"// Create non-contiguous tensor (on dimension {dim_to_change})")
            code.append(f"auto big_tensor = ntt::make_tensor<{element_cpp_type}>({big_shape_expr});")
            code.append(f"NttTest::init_tensor(big_tensor, min_input, max_input);")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view_from_address<{element_cpp_type}>(")
            code.append(f"    big_tensor.elements().data(),")
            code.append(f"    {shape_expr},")
            code.append(f"    big_tensor.strides());")

        return code

    def generate_test_prologue(self, test_suite_prefix, datatype, test_name, P, dim_names, dims, axes=None):
        """generate test function header, constant P and dimension constants"""
        code = [f"TEST({test_suite_prefix}_{datatype.name_suffix}, {test_name}) {{"]
        if P:
            code.append(f"    constexpr size_t P = {P};")

        # define dimension constants
        for i, (name, size) in enumerate(zip(dim_names, dims)):
            if axes and i in axes:
                code.append(f"    constexpr size_t {name}_coefficient = {size};")
                code.append(f"    constexpr size_t {name} = {name}_coefficient * P;")
            else:
                code.append(f"    constexpr size_t {name} = {size};")

        code.extend([f"    {datatype.cpp_type} min_input = {datatype.min_val};",
                     f"    {datatype.cpp_type} max_input = {datatype.max_val};", ""])
        return code

    def generate_reference_and_comparison_code(self,
                                           datatype, continuity, dim_names, shape_type, is_fp8,
                                           input_element_type,
                                           output_element_type,
                                           output_shape_expr,
                                           ort_ref_code,
                                           ntt_output_var_name = "ntt_output1",
                                           ntt_output_var_is_vector = False):
        code = []
        input_dims_expr = [f"{name}" for name in dim_names]

        ort_input_tensor = "ntt_input"
        if not continuity.is_contiguous:
            if is_fp8:
                ort_input_tensor = "ntt_input_uint8"
            else:
                code.append("    // Copy to contiguous tensor for ORT reference")
                code.append(f"    auto continuous_input = ntt::make_tensor<{input_element_type}>({self.generate_shape_init(shape_type, input_dims_expr)});")
                code.append("    ")
                for i, name in enumerate(dim_names):
                    code.append(f"    {'    ' * i}for (size_t {name.lower()} = 0; {name.lower()} < {name}; {name.lower()}++) {{")
                indices = [f"{name.lower()}" for name in dim_names]
                code.append(f"    {'    ' * len(dim_names)}continuous_input({', '.join(indices)}) = ntt_input({', '.join(indices)});")
                for i in range(len(dim_names)-1, -1, -1):
                    code.append(f"    {'    ' * i}}}")
                code.append("")
                ort_input_tensor = "continuous_input"
        elif is_fp8:
            ort_input_tensor = "ntt_input_uint8"

        ort_ref = ort_ref_code
        ort_ref[1] = f"    auto ort_input = NttTest::ntt2ort({ort_input_tensor});"
        code.extend([f"    {line}" for line in ort_ref])
        code.append("")

        code.append("    // Compare results")
        ntt_output_for_comp = ntt_output_var_name
        if is_fp8:
            ntt_output_for_comp += "_uint8"
            output_element_type_uint8 = 'uint8_t'
            if ntt_output_var_is_vector:
                output_element_type_uint8 = output_element_type.replace(datatype.cpp_type, 'uint8_t')

            code.append(f"    auto ntt_output2_uint8 = ntt::make_tensor<{output_element_type_uint8}>({output_shape_expr});")
            code.append(f"    NttTest::ort2ntt(ort_output, ntt_output2_uint8);")
            code.append(f"    EXPECT_TRUE(NttTest::compare_tensor({ntt_output_for_comp}, ntt_output2_uint8));")
        else:
            code.append(f"    auto ntt_output2 = ntt::make_tensor<{output_element_type}>({output_shape_expr});")
            code.append(f"    NttTest::ort2ntt(ort_output, ntt_output2);")
            code.append(f"    EXPECT_TRUE(NttTest::compare_tensor({ntt_output_for_comp}, ntt_output2));")
        
        code.append("}")
        code.append("")
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

    def _build_vector_cpp_type(self, base_cpp_type: str, vector_rank: int, P: Optional[str], axes_count: Optional[int] = None) -> str:
        """Utility: given primitive cpp type, return the full `ntt::vector<..., ...>` expression.
        When ``vector_rank == 0`` it just returns the primitive type.
        When ``vector_rank > 0`` the caller **must** provide ``P`` – the compile-time pack number – and, if ``vector_rank > 1``, also ``axes_count`` (how many axes are packed).
        """
        if vector_rank == 0:
            return base_cpp_type
        if P is None:
            raise ValueError("P must be provided when vector_rank > 0")
        if vector_rank == 1:
            return f"ntt::vector<{base_cpp_type}, {P}>"
        if axes_count is None:
            raise ValueError("axes_count must be provided when vector_rank > 1")
        ps = ", ".join([f"P"] * axes_count)
        return f"ntt::vector<{base_cpp_type}, {ps}>"

    # -------------------------------------------------------------------------
    #  High-level code generation helpers that follow the unified test skeleton
    #  described by the user: NTT-side execution, ORT-side execution, comparison
    # -------------------------------------------------------------------------
    def generate_ntt_input_section(self,
                                   datatype: DataType,
                                   shape_type: str,
                                   dim_names: List[str],
                                   continuity: Continuity,
                                   vector_rank: int = 0,
                                   P: Optional[str] = None,
                                   axes_count: Optional[int] = None,
                                   var_name: str = "ntt_input") -> List[str]:
        """Generate C++ code for *Step-1* — create NTT input tensor according to
        1) scalar / vector, 2) contiguous / non-contiguous. The resulting tensor
        variable will be called ``var_name``.
        """
        comment_lines = ["// ------------------------------------------------------------------",
                         "// 1. create NTT input ",
                         "// ------------------------------------------------------------------"]

        dims_expr = [name for name in dim_names]  # use dimension constants

        # Re-use the existing, well-tested generate_tensor_init helper
        body = self.generate_tensor_init(datatype,
                                         shape_type,
                                         dims_expr,
                                         continuity,
                                         var_name,
                                         vector_rank,
                                         P,
                                         axes_count)
        return comment_lines + body + [""]

    def generate_ntt_operation_section(self,
                                       operation_lines: list[str]) -> list[str]:
        """Wrap actual NTT kernel call with section comment."""
        header = ["// ------------------------------------------------------------------",
                  "// 2. call NTT operation to get NTT output (under test)",
                  "// ------------------------------------------------------------------"]
        return header + operation_lines + [""]

    def generate_ntt_output_and_op_section(self,
                                           datatype: DataType,
                                           output_shape_expr: str,
                                           deal_fp8: int,
                                           ntt_op_call_lines: List[str],
                                           output_var_name: str = "ntt_output1",
                                           output_element_type: Optional[str] = None) -> List[str]:
        """Generates code for creating NTT output tensor, calling the NTT operation,
        and handling FP8 casting.
        """
        if output_element_type is None:
            output_element_type = datatype.cpp_type

        output_tensor_code = [
            f"// Create output tensor",
            f"auto {output_var_name} = ntt::make_tensor<{output_element_type}>({output_shape_expr});",
            ""
        ]
        op_section = output_tensor_code + ntt_op_call_lines
        if deal_fp8 == 1:
            uint8_type = "uint8_t" if "vector" not in output_element_type else output_element_type.replace(datatype.cpp_type, "uint8_t")
            op_section.extend([
                f"auto {output_var_name}_uint8 = ntt::make_tensor<{uint8_type}>({output_shape_expr});",
                f"NttTest::reinterpret_cast_fp8_to_uint8({output_var_name}, {output_var_name}_uint8);",
                ""
            ])

        return self.generate_ntt_operation_section(op_section)

    def generate_ort_input_section(self,
                                   datatype: DataType,
                                   shape_type: str,
                                   dim_names: list[str],
                                   continuity: Continuity,
                                   deal_fp8: int,
                                   P: Optional[str] = None,
                                   vector_rank: int = 0,
                                   axes_count: Optional[int] = None,
                                   ort_input_var_name: str = "ort_input",
                                   ntt_input_var_name: str = "ntt_input") -> list[str]:
        """Generate code for *ORT side* step-1: convert NTT input → ORT input,
        taking care of contiguous copy and fp8 cast when required."""
        lines = ["// ------------------------------------------------------------------",
                 "// 1. build ORT input tensor",
                 "// ------------------------------------------------------------------"]

        input_dims_expr = [name for name in dim_names]

        # Decide which NTT tensor will be fed to ortki
        ort_src_tensor = ntt_input_var_name
        if deal_fp8 == 1:
            # 1.3: if ntt input is fp8, first cast to uint8 tensor.
            # The resulting uint8 tensor is always contiguous.
            input_shape_expr = self.generate_shape_init(shape_type, input_dims_expr)
            uint8_cpp_type = self._build_vector_cpp_type("uint8_t", vector_rank, P, axes_count)
            lines.append(f"    auto {ntt_input_var_name}_uint8 = ntt::make_tensor<{uint8_cpp_type}>({input_shape_expr});")
            lines.append(f"    NttTest::reinterpret_cast_fp8_to_uint8({ntt_input_var_name}, {ntt_input_var_name}_uint8);")
            lines.append(f"")
            ort_src_tensor = f"{ntt_input_var_name}_uint8"
        elif deal_fp8 == 2:
            input_shape_expr = self.generate_shape_init(shape_type, input_dims_expr)
            fp16_cpp_type = self._build_vector_cpp_type("half", vector_rank, P, axes_count)
            lines.append(f"    // Cast fp8 input to fp16 for ORT reference computation")
            lines.append(f"    auto {ntt_input_var_name}_fp16 = ntt::make_tensor<{fp16_cpp_type}>({input_shape_expr});")
            lines.append(f"    ntt::cast({ntt_input_var_name}, {ntt_input_var_name}_fp16);")
            lines.append(f"")
            ort_src_tensor = f"{ntt_input_var_name}_fp16"
        elif not continuity.is_contiguous:
            # 1.2: if not fp8 and non-contiguous, copy to a contiguous buffer.
            # For vector types, the element type is a vector.
            element_cpp_type = self._build_vector_cpp_type(datatype.cpp_type, vector_rank, P, axes_count)
            shape_expr = self.generate_shape_init(shape_type, input_dims_expr)
            lines.append(f"    auto continuous_input = ntt::make_tensor<{element_cpp_type}>({shape_expr});")

            # nested copy loops
            lines.append("")
            for i, name in enumerate(dim_names):
                indent = "    " * i
                lines.append(f"    {indent}for (size_t {name.lower()} = 0; {name.lower()} < {name}; {name.lower()}++) {{")
            indices = ", ".join([n.lower() for n in dim_names])
            lines.append(f"    {'    ' * len(dim_names)}continuous_input({indices}) = {ntt_input_var_name}({indices});")
            for i in range(len(dim_names) - 1, -1, -1):
                indent = "    " * i
                lines.append(f"    {indent}}}")
            lines.append("")
            ort_src_tensor = "continuous_input"

        # At this point, ort_src_tensor is either:
        # 1. The original ntt_input (if contiguous and not fp8)
        # 2. A contiguous copy (if non-contiguous and not fp8)
        # 3. A uint8-casted tensor (if fp8)
        lines.append(f"    auto {ort_input_var_name} = NttTest::ntt2ort({ort_src_tensor});")
        lines.append("")
        return lines

    def generate_ort_operation_section(self, ort_operation_lines: list[str]) -> list[str]:
        """Wrap ortki kernel invocation section."""
        header = ["// ------------------------------------------------------------------",
                  "// 2. call ortki kernel to generate ORT output",
                  "// ------------------------------------------------------------------"]
        return header + ort_operation_lines + [""]

    def generate_ort_back2ntt_and_compare_section(self,
                                                  datatype: DataType,
                                                  output_element_cpp_type: str,
                                                  output_shape_expr: str,
                                                  deal_fp8: int,
                                                  ntt_output_var_name: str = "ntt_output1",
                                                  ort_output_var_name: str = "ort_output") -> list[str]:
        """Generate code to convert ORT output back to NTT tensor (golden) and
        compare with tested NTT output."""
        lines = ["// ------------------------------------------------------------------",
                 "// 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output",
                 "// ------------------------------------------------------------------"]
        
        if deal_fp8 == 0:  # Not fp8
            golden_var_name = "ntt_golden"
            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_var_name}, {golden_var_name}));")
        elif deal_fp8 == 1:  # fp8 with uint8 comparison
            ntt_output_to_compare = f"{ntt_output_var_name}_uint8"
            golden_var_name = "ntt_golden_uint8"
            golden_cpp_type = "uint8_t" if "vector" not in output_element_cpp_type else output_element_cpp_type.replace(datatype.cpp_type, "uint8_t")

            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{golden_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_to_compare}, {golden_var_name}));")
        elif deal_fp8 == 2:  # fp8 with fp16 intermediate, compare fp8
            golden_fp16_var_name = "ntt_golden_fp16"
            golden_fp16_cpp_type = output_element_cpp_type.replace(datatype.cpp_type, "half")
            
            lines.append(f"// Golden output is in fp16, cast it back to fp8 for comparison")
            lines.append(f"auto {golden_fp16_var_name} = ntt::make_tensor<{golden_fp16_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_fp16_var_name});")

            golden_fp8_var_name = "ntt_golden_fp8"
            lines.append(f"auto {golden_fp8_var_name} = ntt::make_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"ntt::cast({golden_fp16_var_name}, {golden_fp8_var_name});")

            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_var_name}, {golden_fp8_var_name}));")

        lines.append("}")
        lines.append("")
        return lines

def generate_cmake_list(directory, filenames, output_filename, variable_name):
    """generate a .cmake file that contains the list of generated test files"""
    cmake_list_path = os.path.join(directory, output_filename)
    with open(cmake_list_path, "w") as f:
        f.write(f"# This file is generated automatically. DO NOT EDIT.\n")
        f.write(f"set({variable_name}\n")
        for name in filenames:
            f.write(f"    ${{CMAKE_CURRENT_LIST_DIR}}/{name}\n") # use relative path to current CMakeLists.txt
        f.write(")\n")
    print(f"Generated CMake list: {cmake_list_path}")
