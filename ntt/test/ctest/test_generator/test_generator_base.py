#!/usr/bin/env python3
"""
Base classes and utilities for generating NTT test cases.
"""

import os
from collections import namedtuple
from typing import List, Optional
from enum import Enum


# is_contiguous: bool
# non_contiguous_dim: int or None
# big_tensor_op: str or None - How to build the big tensor at given non_contiguous_dim
Continuity = namedtuple('Continuity', ['is_contiguous', 'non_contiguous_dim', 'big_tensor_op'])
DataType = namedtuple('DataType', ['cpp_type', 'name_suffix', 'min_val', 'max_val', 'integer_only'])

class ShapeType(Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"

    @classmethod
    def from_input(cls, value):
        """
        A factory method to create a ShapeType enum instance from a string or boolean.
        """
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                raise ValueError(f"Invalid shape_type string: '{value}'. Must be 'fixed' or 'dynamic'.")
        elif isinstance(value, bool):
            return cls.DYNAMIC if value else cls.FIXED
        elif isinstance(value, cls):
            return value
        else:
            raise TypeError(f"Unsupported shape_type type: {type(value)}. Must be str, bool, or ShapeType.")

    def is_dynamic(self):
        return self == ShapeType.DYNAMIC

    def is_fixed(self):
        return self == ShapeType.FIXED

ALL_DATATYPES = [
    DataType('bool', 'Bool', 'false', 'true', False),
    DataType('uint8_t', 'Uint8', '0', '16', True),
    DataType('uint16_t', 'Uint16', '0', '256', True),
    DataType('uint32_t', 'Uint32', '0', '15536', True),
    DataType('uint64_t', 'Uint64', '0', '1000000', True),
    DataType('int8_t', 'Int8', '-11', '11', True),
    DataType('int16_t', 'Int16', '-181', '181', True),
    DataType('int32_t', 'Int32', '-32761', '32761', True),
    DataType('int64_t', 'Int64', '-1000000', '1000000', True),
    DataType('half', 'Float16', 'half(-100.0f)', 'half(100.0f)', False),
    DataType('float', 'Float32', '-3.4e15', '3.4e15', False),
    DataType('double', 'Float64', '-1.7e150', '1.7e150', False),
    DataType('bfloat16', 'Bfloat16', '-1.0e10_bf16', '1.0e10_bf16', False),
    DataType('float_e4m3_t', 'Float8e4m3', 'float_e4m3_t(-16.0f)', 'float_e4m3_t(16.0f)', False),
    DataType('float_e5m2_t', 'Float8e5m2', 'float_e5m2_t(-32.0f)', 'float_e5m2_t(32.0f)', False)
]

class BaseTestGenerator:
    def __init__(self):
        self.test_cases = []
        self.ort_datatype_map = {
            'bool': 'DataType_BOOL',
            'uint8_t': 'DataType_UINT8',
            'uint16_t': 'DataType_UINT16', 
            'uint32_t': 'DataType_UINT32',
            'uint64_t': 'DataType_UINT64',
            'int8_t': 'DataType_INT8',
            'int16_t': 'DataType_INT16',
            'int32_t': 'DataType_INT32',
            'int64_t': 'DataType_INT64',
            'half': 'DataType_FLOAT16',
            'float': 'DataType_FLOAT',
            'double': 'DataType_DOUBLE',
            'bfloat16': 'DataType_BFLOAT16',
        }
        self.indent = "    "
        self.simple_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            # Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="+3"),
        ]

        self.ulp_tolerances = {
            "default": {
                "default": 1
            }
        }

        self.ort_custom_function = {}
        self.ntt_op_str = ""


    def _generate_ort_custom_op(self, datatype, custom_op_name):
        """Generate custom ORT operation functions"""
        if custom_op_name in self.ort_custom_function:
            return self.ort_custom_function[custom_op_name](datatype)
        return ""

    def _get_ulp_tolerance(self, op_str, datatype):
        """Get the ULP tolerance for a specific operation and data type."""
        if op_str in self.ulp_tolerances:
            return self.ulp_tolerances[op_str].get(datatype.cpp_type, self.ulp_tolerances[op_str]["default"])
        return self.ulp_tolerances["default"].get(datatype.cpp_type, self.ulp_tolerances["default"]["default"])

    def _need_cast_in_ort(self, datatype, op_str):
        """Check if datatype needs to be cast in ORT for the given operation."""
        if not hasattr(self, 'types_need_cast_in_ort'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement 'types_need_cast_in_ort' attribute"
            )
        types_to_cast_in_ort = self.types_need_cast_in_ort.get(op_str, self.types_need_cast_in_ort["default"])
        return datatype.cpp_type in types_to_cast_in_ort

    def get_unpacked_dims(self, dim_names, unpack_axes) -> List[str]:
        """Generate dimension expressions for an unpack operation."""
        output_dims = []
        ndim = len(dim_names)
        positive_unpack_axes = [ax if ax >= 0 else ndim + ax for ax in unpack_axes]
        for i, name in enumerate(dim_names):
            param = ""
            if i == positive_unpack_axes[-1]:
                param = "P"
            else:
                param = "4"
            if i in positive_unpack_axes:
                output_dims.append(f"{name} * {param}")
            else:
                output_dims.append(name)
        return output_dims

    def generate_shape_init(self, shape_type, dim_spec):
        shape_type = ShapeType.from_input(shape_type)
        if shape_type.is_fixed():
            dim_strs = [f"{d}" for d in dim_spec]
            return f"ntt::fixed_shape_v<{', '.join(dim_strs)}>"
        else:  # dynamic
            dim_strs = [str(d) for d in dim_spec]
            return f"ntt::make_shape({', '.join(dim_strs)})"

    def _get_allow_zr(self, var_name ):
        ans = "true"
        if(self.is_div_operation() and "rhs" in var_name):
            ans = "false"
        if(self.ntt_op_str == "pow" and "lhs" in var_name):
            ans = "false"

        return ans

#shape_type: str: "dynamic" or "fixed"
#shape_type: bool: True (is_dynamic) or False (is_fixed)
#dim_spec: dim_names(list[str]) or dim_spec(list[int])
    def generate_tensor_init(self, datatype, shape_type,
                             dim_spec, continuity,
                             vector_rank, var_name, name_suffix, P=None, integer_only=False):
        code = []
        shape_expr = self.generate_shape_init(shape_type, dim_spec)
        element_cpp_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, P)

        if continuity.is_contiguous:
            code.append(f"auto {var_name} = ntt::make_tensor<{element_cpp_type}>({shape_expr});")
            allow_zr = self._get_allow_zr(var_name)
            integer_only_str = "true" if integer_only else "false"
            code.append(f"NttTest::init_tensor({var_name}, {datatype.min_val}, {datatype.max_val}, {allow_zr}, {integer_only_str});")
        else:  # non-contiguous
            big_dims = dim_spec.copy()
            dim_to_change = continuity.non_contiguous_dim
            op = continuity.big_tensor_op

            if dim_to_change is not None and op is not None and dim_to_change < len(big_dims):
                big_dims[dim_to_change] = f"({big_dims[dim_to_change]}) {op}"

            big_shape_expr = self.generate_shape_init(shape_type, big_dims)

            code.append(f"// Create non-contiguous tensor (on dimension {dim_to_change})")
            code.append(f"auto big_tensor{name_suffix} = ntt::make_tensor<{element_cpp_type}>({big_shape_expr});")
            allow_zr = self._get_allow_zr(var_name)
            integer_only_str = "true" if integer_only else "false"
            code.append(f"NttTest::init_tensor(big_tensor{name_suffix}, {datatype.min_val}, {datatype.max_val}, {allow_zr}, {integer_only_str});")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view_from_address<{element_cpp_type}>(")
            code.append(f"    big_tensor{name_suffix}.elements().data(),")
            code.append(f"    {shape_expr},")
            # Use canonicalized strides so that any dimension with size 1 gets stride 0.
            # This ensures correct semantics when the view shape has broadcastable dims.
            code.append(f"    {self._build_view_strides(shape_expr, f'big_tensor{name_suffix}.strides()')}")
            code.append(f"    );")

        return code

    def _build_view_strides(self, shape_expr: str, base_strides_expr: str) -> str:
        """
        Build the strides expression for a tensor view. Any dimension with size 1
        in the target view shape must have a stride of 0 to support broadcasting semantics.

        We achieve this by invoking `ntt::canonicalize_strides(view_shape, base_strides)`
        in the generated C++.
        """
        return f"ntt::canonicalize_strides({shape_expr}, {base_strides_expr})"


    def generate_demension_constants(self, dim_names, dims, datatype, P):
        code = []
        if P is not None:
            code.append(f"    constexpr size_t P = {P};")

        for i, (name, size) in enumerate(zip(dim_names, dims)):
           code.append(f"    constexpr size_t {name} = {size};")
        return code

    def generate_function_name(self, test_suite_prefix, datatype, test_name):
        code = [f"TEST({test_suite_prefix}_{datatype.name_suffix}, {test_name}) {{"]
        return code

    def generate_test_prologue(self, test_suite_prefix, datatype, test_name, P, dim_names, dims, pack_axes=None):
        """generate test function header, constant P and dimension constants"""
        code = [f"TEST({test_suite_prefix}_{datatype.name_suffix}, {test_name}) {{"]
        if (P and (pack_axes is not None)) or ("unpack" in test_name):
            code.append(f"    constexpr size_t P = {P};")

        # define dimension constants
        for i, (name, size) in enumerate(zip(dim_names, dims)):
            if pack_axes and (i in pack_axes):
                vec_param = "P" if i  == pack_axes[-1] else "4"
                code.append(f"    constexpr size_t {name}_coefficient = {size};")
                code.append(f"    constexpr size_t {name} = {name}_coefficient * {vec_param};")
            else:
                code.append(f"    constexpr size_t {name} = {size};")

        # code.extend([f"    {datatype.cpp_type} min_input = {datatype.min_val};",
        #              f"    {datatype.cpp_type} max_input = {datatype.max_val};", ""])
        return code

    # def generate_min_max_constants(self, datatype):
    #     code = []
    #     # code.append(f"    {datatype.cpp_type} min_input = {datatype.min_val};")
    #     # code.append(f"    {datatype.cpp_type} max_input = {datatype.max_val};")
    #     return code


    def generate_copy_to_contiguous_code(self, input_element_type, shape_type, dim_names, input_var_name="ntt_input", output_var_name="continuous_input"):
        code = []
        input_dims_expr = [f"{name}" for name in dim_names]
        code.append("    // Copy to contiguous tensor for ORT reference")
        code.append(f"    auto {output_var_name} = ntt::make_unique_tensor<{input_element_type}>({self.generate_shape_init(shape_type, input_dims_expr)});")
        code.append("    ")

        output_var_name = f"*{output_var_name}"

        iter_var_names = ["i", "j", "k", "l", "m"]
        for i, name in enumerate(dim_names):
            code.append(f"    {'    ' * i}for (size_t {iter_var_names[i]} = 0; {iter_var_names[i]} < {name}; {iter_var_names[i]}++) {{")
        indices = [f"{iter_var_names[i]}" for i in range(len(dim_names))]
        code.append(f"    {'    ' * len(dim_names)}({output_var_name})({', '.join(indices)}) = {input_var_name}({', '.join(indices)});")
        for i in range(len(dim_names) - 1, -1, -1):
            code.append(f"    {'    ' * i}}}")
        code.append("")
        return code, output_var_name

    def generate_pack_axes_str(self, axes):
        if len(axes) == 1:
            return f"ntt::fixed_shape_v<{axes[0]}>"
        else:
            return f"ntt::fixed_shape_v<{', '.join(map(str, axes))}>"

    def generate_reference_and_comparison_code(self,
                                           datatype, continuity, dim_names, shape_type, is_fp8,
                                           input_element_type,
                                           output_element_type,
                                           output_shape_expr,
                                           ort_ref_code,
                                           ntt_output_var_name = "ntt_output1",
                                           ntt_output_var_is_vector = False):
        code = []
        ort_input_tensor = "ntt_input"
        if not continuity.is_contiguous:
            if is_fp8:
                ort_input_tensor = "ntt_input_uint8"
            else:
                copy_code, ort_input_tensor = self.generate_copy_to_contiguous_code(input_element_type, shape_type, dim_names)
                code.extend(copy_code)
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

    def generate_P_constants(self, P_val):
        code = []
        code.append(f"    constexpr size_t P = {P_val};")
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

    def is_div_operation(self) -> bool:
        """Override in subclasses to indicate whether current operation is division.
        Returns True for div operations to disable allow_zr in init_tensor.
        """
        return False

    def get_element_cpp_type(self, base_cpp_type: str, vector_rank: int, P: Optional[str]) -> str:
        """Utility: given primitive cpp type, return the full `ntt::vector<..., ...>` expression.
        When ``vector_rank == 0`` it just returns the primitive type.
        When ``vector_rank > 0`` the caller **must** provide ``P`` – the compile-time pack number.
        P can be a single value or a tuple/list for multi-dimensional vectors.
        """

        if vector_rank == 0:
            return base_cpp_type
        if P is None:
            raise ValueError("P must be provided when vector_rank > 0")
        
        # Handle tuple/list case for multi-dimensional vectors
        if isinstance(P, (tuple, list)):
            if len(P) != vector_rank:
                raise ValueError("Length of P tuple/list must match vector_rank")
            # Convert each element to string, using "P" for the last element
            ps_list = []
            for i, p in enumerate(P):
                if i == len(P) - 1:
                    ps_list.append("P" if p is None else str(p))
                else:
                    ps_list.append("4" if p is None else str(p))
            ps = ", ".join(ps_list)
        else:
            # Original behavior for single P value
            if vector_rank == 1:
                ps = P
            elif vector_rank > 1:
                ps = ", ".join([f"4"] * (vector_rank-1)) + ", " + P

        return f"ntt::vector<{base_cpp_type}, {ps}>"

    # -------------------------------------------------------------------------
    #  High-level code generation helpers that follow the unified test skeleton
    #  described by the user: NTT-side execution, ORT-side execution, comparison
    # -------------------------------------------------------------------------
    def generate_ntt_input_section(self,
                                   datatype: DataType,
                                   shape_type: str,
                                   dims_spec,
                                   continuity: Continuity,
                                   vector_rank: int = 0,
                                   P: Optional[str] = None,
                                   var_name: str = "ntt_input",
                                   name_suffix: str = "") -> List[str]:
        """Generate C++ code for *Step-1* — create NTT input tensor according to
        1) scalar / vector, 2) contiguous / non-contiguous. The resulting tensor
        variable will be called ``var_name``.
        """
        comment_lines = ["// ------------------------------------------------------------------",
                         "// 1. create NTT input ",
                         "// ------------------------------------------------------------------"]


        # Re-use the existing, well-tested generate_tensor_init helper
        body = self.generate_tensor_init(datatype,
                                         shape_type,
                                         dims_spec,
                                         continuity,
                                         vector_rank,
                                         var_name,
                                         name_suffix,
                                         P)
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
                                           cast_mode: int,
                                           ntt_op_call_lines: List[str],
                                           output_var_name: str = "ntt_output1",
                                           output_element_type = None) -> List[str]:
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
        if cast_mode == 1:
            uint8_type = "uint8_t" if "vector" not in output_element_type else output_element_type.replace(datatype.cpp_type, "uint8_t")
            op_section.extend([
                f"auto {output_var_name}_uint8 = ntt::make_tensor<{uint8_type}>({output_shape_expr});",
                f"NttTest::reinterpret_cast_fp8_to_uint8({output_var_name}, {output_var_name}_uint8);",
                ""
            ])

        return self.generate_ntt_operation_section(op_section)
    
    def generate_ort_cast_back(self, datatype, cast_input_var = "ort_output", cast_output_var = "ort_golden"):
        """cast ort tensor of double back to ort tensor of original type"""
        """if original type is uint, cast to int64 first"""
        code = []
        code.append(f"// Cast outputs from double to original datatype")
        original_type = self.ort_datatype_map[datatype.cpp_type]
        var_name_cast_to_orig_type = cast_input_var
        if("uint" in datatype.cpp_type):
            var_name_cast_to_orig_type = cast_output_var+"int"
            code.append(f"auto {var_name_cast_to_orig_type} = ortki_Cast({cast_input_var}, 1, ortki::DataType_INT64);")
        code.append(f"auto {cast_output_var} = ortki_Cast({var_name_cast_to_orig_type}, 1, ortki::{original_type});")
        return code

    # back2ntt used in ntt cast version.
    def _cast_ort_golden_double_into_ntt_shape(self, code, datatype, ntt_op_str,
            output_is_dynamic, output_dims_spec, output_vector_rank,
            output_vec_param, ort_golden_double_var, cast_target = "double"):
        """Process ORT output back to NTT format with proper casting and vectorization"""
        """
        5. transform back to ntt tensor of double scalar
        6. cast back to original type, still tensor of scalar
        7. vectorized back to original tensor of vector (if necessary)
        """
        # 3. transform ort_golden_double to ntt_golden{cpp_type}_scalar
        code.append(f"//  transform ort_golden_{cast_target} to ntt_golden{datatype.cpp_type}_scalar")

        
        # # Get shape of ntt_golden_double_scalar based on aligned shapes and operation
        golden_scalar_dims = self.generate_ntt_golden_double_scalar_dims_spec(ntt_op_str, output_dims_spec, output_vector_rank)

        golden_scalar_shape_expr = self.generate_shape_init(output_is_dynamic, golden_scalar_dims)

        code.append(f"auto ntt_golden_{cast_target}_scalar = ntt::make_unique_tensor<{cast_target}>({golden_scalar_shape_expr});")
        code.append(f"NttTest::ort2ntt({ort_golden_double_var}, *ntt_golden_{cast_target}_scalar);")
        code.append("")
        code.append(f"auto ntt_golden_{datatype.cpp_type}_scalar = ntt::make_unique_tensor<{datatype.cpp_type}>({golden_scalar_shape_expr});")
        code.append(f"ntt::cast(*ntt_golden_{cast_target}_scalar, *ntt_golden_{datatype.cpp_type}_scalar);")

        # 4. transform ntt_golden_{datatype.cpp_type}_scalar to ntt_golden
        code.append("")
        if output_vector_rank > 0:
            # 4.b if ntt_output is tensor of vector
            code.append("// 4.b if ntt_output is tensor of vector")
            unsqueeze_shape_dims = output_dims_spec + (["1"] if output_vector_rank == 1 else ["1", "1"])
            unsqueeze_shape_expr = self.generate_shape_init(output_is_dynamic, unsqueeze_shape_dims)
            vector_type_str = self.get_element_cpp_type(datatype.cpp_type, output_vector_rank, output_vec_param)
            code.append(f"auto ntt_golden_unsqueeze = ntt::make_tensor<{vector_type_str}>({unsqueeze_shape_expr});")
            dims_spec_len = len(output_dims_spec)
            pack_dims = f"{dims_spec_len}" if output_vector_rank == 1 else f"{dims_spec_len}, {dims_spec_len + 1}"
            code.append(f"ntt::pack(*ntt_golden_{datatype.cpp_type}_scalar, ntt_golden_unsqueeze, fixed_shape_v<{pack_dims}>);")
            code.append(f"auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<{pack_dims}>));")
        else:
            # 4.a if ntt_output is not tensor of vector
            code.append("// 4.a if ntt_output is not tensor of vector")
            code.append(f"auto ntt_golden = *ntt_golden_{datatype.cpp_type}_scalar;")

        code.append("")



    def generate_ntt_golden_double_scalar_dims_spec(self, ntt_op_str, output_dims_spec, output_vector_rank):
        if output_vector_rank > 0:
            if output_vector_rank == 1:
                golden_scalar_dims = output_dims_spec + ["P"]
            else:  # 2D vector
                if ntt_op_str == "outer_product":
                    golden_scalar_dims = output_dims_spec  + ["P", "P"]
                else:
                    golden_scalar_dims = output_dims_spec + ["4", "P"]
        else:
            golden_scalar_dims = output_dims_spec

        return golden_scalar_dims
    
    def generate_ort_output(self, datatype, ntt_op_str):
        
        # Check both dictionaries for the operation string
        if ntt_op_str in self.op_str_map_exhaustive:
            op_str = self.op_str_map_exhaustive[ntt_op_str]
        elif ntt_op_str in self.op_str_map_simplified:
            op_str = self.op_str_map_simplified[ntt_op_str]
        else:
            raise KeyError(f"Operation '{ntt_op_str}' not found in either op_str_map_exhaustive or op_str_map_simplified")
            
        if callable(op_str):
            op_str = op_str(datatype)
        return [
            f"// Execute Ort operation",
            f"{op_str}",
            ""
        ]

    def _prepare_contiguous_input(self, input_name, datatype, vector_rank, vec_param, 
                                  is_dynamic_shape, dims_spec, continuity):
        
        continuity_var_name = input_name
        element_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, vec_param)
        code = []
        
        if not continuity.is_contiguous:
            continuity_var_name = f"{input_name}_contiguous"
            copy_code, _ = self.generate_copy_to_contiguous_code(
                element_type,
                is_dynamic_shape,
                dims_spec,
                input_name,
                continuity_var_name
            )
            continuity_var_name = f"*{continuity_var_name}"
            code.extend(copy_code)
        
        return continuity_var_name, code

    """ Not a good abstraction """
    def generate_ort_input_section(self,
                                   datatype: DataType,
                                   shape_type,
                                   dims_spec,
                                   continuity,
                                   cast_mode: int,
                                   P: Optional[str] = None,
                                   vector_rank: int = 0,
                                   ort_input_var_name: str = "ort_input",
                                   ntt_input_var_name: str = "ntt_input",
                                   name_suffix: str = "") -> list[str]:
        """Generate code for *ORT side* step-1: convert NTT input → ORT input,
        taking care of contiguous copy and fp8 cast when required."""
        lines = ["// ------------------------------------------------------------------",
                 "// 1. build ORT input tensor",
                 "// ------------------------------------------------------------------"]

        # Decide which NTT tensor will be fed to ortki
        ort_src_tensor = ntt_input_var_name
        if cast_mode == 1:
            # For cast, if ntt input is fp8, first cast to uint8 tensor.
            # The resulting uint8 tensor is always contiguous.
            input_shape_expr = self.generate_shape_init(shape_type, dims_spec)
            uint8_cpp_type = self.get_element_cpp_type("uint8_t", vector_rank, P)
            lines.append(f"    auto {ntt_input_var_name}_uint8 = ntt::make_tensor<{uint8_cpp_type}>({input_shape_expr});")
            lines.append(f"    NttTest::reinterpret_cast_fp8_to_uint8({ntt_input_var_name}, {ntt_input_var_name}_uint8);")
            lines.append(f"")
            ort_src_tensor = f"{ntt_input_var_name}_uint8"
        elif not continuity.is_contiguous:
            # 1.2: if not fp8 and non-contiguous, copy to a contiguous buffer.
            # For vector types, the element type is a vector.
            element_cpp_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, P)
            shape_expr = self.generate_shape_init(shape_type, dims_spec)
            lines.append(f"  auto continuous_input{name_suffix} = ntt::make_tensor<{element_cpp_type}>({shape_expr});")

            iter_var_names = ["i", "j", "k", "l", "m"]
            # nested copy loops
            lines.append("")
            for i, iter_end in enumerate(dims_spec):
                indent = "    " * i
                lines.append(f"    {indent}for (size_t {iter_var_names[i]} = 0; {iter_var_names[i]} < {iter_end}; {iter_var_names[i]}++) {{")
            indices = ", ".join([iter_var_names[i] for i in range(len(dims_spec))])
            lines.append(f"    {'    ' * len(dims_spec)}continuous_input{name_suffix}({indices}) = {ntt_input_var_name}({indices});")
            for i in range(len(dims_spec) - 1, -1, -1):
                indent = "    " * i
                lines.append(f"    {indent}}}")
            lines.append("")
            ort_src_tensor = f"continuous_input{name_suffix}"

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
    

    #back2ntt early version
    def generate_ort_back2ntt_and_compare_section(self,
                                                  datatype: DataType,
                                                  output_element_cpp_type: str,
                                                  output_shape_expr: str,
                                                  cast_mode: int,
                                                  ntt_output_var_name: str = "ntt_output1",
                                                  ort_output_var_name: str = "ort_output",
                                                  ort_type: str = "double") -> list[str]:
        """Generate code to convert ORT output back to NTT tensor (golden) and
        compare with tested NTT output."""
        lines = ["// ------------------------------------------------------------------",
                 "// 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output",
                 "// ------------------------------------------------------------------"]
        
        if cast_mode == 0:  #  no cast
            golden_var_name = "ntt_golden"
            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_var_name}, {golden_var_name}));")
        elif cast_mode == 1:  # fp8 with uint8 comparison
            ntt_output_to_compare = f"{ntt_output_var_name}_uint8"
            golden_var_name = "ntt_golden_uint8"
            golden_cpp_type = "uint8_t" if "vector" not in output_element_cpp_type else output_element_cpp_type.replace(datatype.cpp_type, "uint8_t")

            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{golden_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_to_compare}, {golden_var_name}));")
        elif cast_mode == 2:  #  cast in ntt
            # ort_type means data type(float, int, double,...)of ort tensor
            golden_ntt_in_ort_type_var = f"ntt_golden_{ort_type}"
            golden_cpp_type = output_element_cpp_type.replace(datatype.cpp_type, ort_type)

            lines.append(f"// Golden output is in ort_type, cast it back to datatype.cpp_type for comparison")
            lines.append(f"auto {golden_ntt_in_ort_type_var} = ntt::make_unique_tensor<{golden_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, *{golden_ntt_in_ort_type_var});")

            golden_signed_int_var = "ntt_golden_signed_int"
            if datatype.cpp_type in ["uint8_t", "uint16_t", "uint32_t", "uint64_t"]:
                int_tensor_cpp_type = output_element_cpp_type.replace(datatype.cpp_type, "int64_t")
                lines.append(f"auto {golden_signed_int_var} = ntt::make_unique_tensor<{int_tensor_cpp_type}>({output_shape_expr});")
                lines.append(f"ntt::cast(*{golden_ntt_in_ort_type_var}, *{golden_signed_int_var});")
                golden_cast_source_var = golden_signed_int_var
            else:
                golden_cast_source_var = golden_ntt_in_ort_type_var
            
            golden_origin_var = "ntt_golden"
            lines.append(f"auto {golden_origin_var} = ntt::make_unique_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"ntt::cast(*{golden_cast_source_var}, *{golden_origin_var});")

            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_var_name}, *{golden_origin_var}));")
        elif cast_mode == 4:   # cast in ort
            lines.append(f"EXPECT_TRUE(NttTest::compare_tensor( ntt_output, ntt_golden));")

        lines.append("}")
        lines.append("")
        return lines

    #back2ntt used for cast in ort 
    def generate_ort_back2ntt(self,
                                datatype: DataType,
                                output_element_cpp_type: str,
                                output_shape_expr: str,
                                cast_mode: int,
                                ntt_output_var_name: str = "ntt_output1",
                                ort_output_var_name: str = "ort_output",
                                ort_type: str = "double"):


        """Generate code to convert ORT output back to NTT tensor (golden)"""
        lines = ["// ------------------------------------------------------------------",
                 "// 3. convert ORT output back to NTT tensor (golden) ",
                 "// ------------------------------------------------------------------"]
        golden_var_name = "ntt_golden"  # Default name if no cast_mode
        if cast_mode == 0:  #  no cast
            golden_var_name = "ntt_golden"
            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
        elif cast_mode == 1:  # fp8 with uint8 comparison
            ntt_output_to_compare = f"{ntt_output_var_name}_uint8"
            golden_var_name = "ntt_golden_uint8"
            golden_cpp_type = "uint8_t" if "vector" not in output_element_cpp_type else output_element_cpp_type.replace(datatype.cpp_type, "uint8_t")

            lines.append(f"auto {golden_var_name} = ntt::make_tensor<{golden_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, {golden_var_name});")
        elif cast_mode == 2:  #  cast in ntt
            golden_ntt_in_ort_type_var = f"ntt_golden_{ort_type}"
            golden_cpp_type = output_element_cpp_type.replace(datatype.cpp_type, ort_type)

            lines.append(f"// Golden output is in ort_type, cast it back to datatype.cpp_type for comparison")
            lines.append(f"auto {golden_ntt_in_ort_type_var} = ntt::make_unique_tensor<{golden_cpp_type}>({output_shape_expr});")
            lines.append(f"NttTest::ort2ntt({ort_output_var_name}, *{golden_ntt_in_ort_type_var});")

            golden_signed_int_var = "ntt_golden_signed_int"
            if datatype.cpp_type in ["uint8_t", "uint16_t", "uint32_t", "uint64_t"]:
                int_tensor_cpp_type = output_element_cpp_type.replace(datatype.cpp_type, "int64_t")
                lines.append(f"auto {golden_signed_int_var} = ntt::make_unique_tensor<{int_tensor_cpp_type}>({output_shape_expr});")
                lines.append(f"ntt::cast(*{golden_ntt_in_ort_type_var}, *{golden_signed_int_var});")
                golden_cast_source_var = golden_signed_int_var
            else:
                golden_cast_source_var = golden_ntt_in_ort_type_var
            
            golden_var_name = "*ntt_golden"
            lines.append(f"auto {generate_var_name} = ntt::make_unique_tensor<{output_element_cpp_type}>({output_shape_expr});")
            lines.append(f"ntt::cast(*{golden_cast_source_var}, {golden_var_name});")


        return lines, golden_var_name

    def generate_compare(
        self,
        ntt_output_var_name: str = "ntt_output",
        golden_var_name: str = "ntt_golden",
        ulp_tolerances = 1
    ) -> list[str]:
        """Generate comparison code between NTT output and golden output."""
        lines =[]
        lines.append(f"EXPECT_TRUE(NttTest::compare_tensor({ntt_output_var_name}, {golden_var_name}, {ulp_tolerances}));")
        lines.append("}")
        lines.append("")
        return lines


def get_numeric_value(value_str: str, cpp_type: str) -> float:
    """Extract numeric value from string representation based on cpp_type"""
    if cpp_type == 'bool':
        return 1.0 if value_str == 'true' else 0.0
    elif cpp_type in ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t']:
        return float(value_str)
    elif cpp_type == 'half':
        # Extract value from half(-100.0f) format
        if value_str.startswith('half(') and value_str.endswith(')'):
            inner = value_str[5:-1]  # Remove 'half(' and ')'
            if inner.endswith('f'):
                inner = inner[:-1]  # Remove 'f'
            return float(inner)
        return float(value_str)
    elif cpp_type == 'float':
        if value_str.endswith('f'):
            return float(value_str[:-1])
        return float(value_str)
    elif cpp_type == 'double':
        return float(value_str)
    elif cpp_type == 'bfloat16':
        # Extract value from -1.0e10_bf16 format
        if value_str.endswith('_bf16'):
            return float(value_str[:-6])  # Remove '_bf16'
        return float(value_str)
    elif cpp_type in ['float_e4m3_t', 'float_e5m2_t']:
        # Extract value from float_e4m3_t(-16.0f) format
        if value_str.startswith(cpp_type + '(') and value_str.endswith(')'):
            inner = value_str[len(cpp_type)+1:-1]  # Remove type and parentheses
            if inner.endswith('f'):
                inner = inner[:-1]  # Remove 'f'
            return float(inner)
        return float(value_str)
    else:
        return float(value_str)

def format_value_for_type(value: float, cpp_type: str) -> str:
    """Format numeric value back to string representation for given cpp_type"""
    if cpp_type == 'bool':
        return 'true' if value > 0.5 else 'false'
    elif cpp_type in ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t']:
        return str(int(value))
    elif cpp_type == 'half':
        return f'half({value}f)'
    elif cpp_type == 'float':
        return f'{value}f'
    elif cpp_type == 'double':
        return str(value)
    elif cpp_type == 'bfloat16':
        return f'{value}_bf16'
    elif cpp_type in ['float_e4m3_t', 'float_e5m2_t']:
        return f'{cpp_type}({value}f)'
    else:
        return str(value)

def clamp_value_strings(from_type: DataType, to_type: DataType) -> tuple[str, str]:
    """
    Clamp the min/max values of from_type to the range of to_type.
    Returns (clamped_min_str, clamped_max_str) in the format of to_type.
    
    Args:
        from_type: Source data type with min_val and max_val as strings
        to_type: Target data type with min_val and max_val as strings
        
    Returns:
        Tuple of (min_value_str, max_value_str) formatted for to_type
    """
    # Get numeric values
    from_min = get_numeric_value(from_type.min_val, from_type.cpp_type)
    from_max = get_numeric_value(from_type.max_val, from_type.cpp_type)
    to_min = get_numeric_value(to_type.min_val, to_type.cpp_type)
    to_max = get_numeric_value(to_type.max_val, to_type.cpp_type)
    
    # Clamp the values
    clamped_min = max(from_min, to_min)
    clamped_max = min(from_max, to_max)
    
    # Format back to strings for to_type
    clamped_min_str = format_value_for_type(clamped_min, from_type.cpp_type)
    clamped_max_str = format_value_for_type(clamped_max, from_type.cpp_type)
    
    return clamped_min_str, clamped_max_str

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
