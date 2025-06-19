#!/usr/bin/env python3
"""
Generate test cases for NTT pack operations
Covering the following cases:
1. Shape types: fixed/dynamic
2. Vector dimensions: 1D/2D
3. Tensor continuity: contiguous/non-contiguous
4. Pack axes: different dimensions
"""



import itertools
from typing import List, Tuple
from collections import namedtuple
import os


# is_contiguous: bool 
# non_contiguous_dim: int or None 
# big_tensor_op: str or None -  How to build the big tensor at given non_contiguous_dim
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

class PackTestGenerator:
    def __init__(self):
        self.test_cases = []
        
    def generate_test_name(self, datatype, shape_type, vector_dim, continuity: Continuity, pack_axis_str, ndim):
        parts = []
        parts.append(datatype.name_suffix)
        parts.append(shape_type)
        parts.append(f"{vector_dim}D_vector")
        
        if continuity.is_contiguous:
            parts.append("contiguous")
        else:
            op_str = "mul2" if continuity.big_tensor_op == "*2" else "add5"
            parts.append(f"non_contiguous_dim{continuity.non_contiguous_dim}_{op_str}")

        parts.append(f"pack_axis_{pack_axis_str}")
        parts.append(f"{ndim}D")
        return "_".join(parts)
    
    def generate_shape_init(self, shape_type, dims):
        if shape_type == "fixed":
            dim_strs = [f"{d}" for d in dims]
            return f"ntt::fixed_shape_v<{', '.join(dim_strs)}>"
        else:  # dynamic
            dim_strs = [str(d) for d in dims]
            return f"ntt::make_shape({', '.join(dim_strs)})"
    
    def generate_tensor_init(self, datatype, shape_type, dims, continuity, var_name):
        code = []
        shape_expr = self.generate_shape_init(shape_type, dims)
        
        if continuity.is_contiguous:
            code.append(f"alignas(32) auto {var_name} = ntt::make_tensor<{datatype.cpp_type}>({shape_expr});")
            code.append(f"NttTest::init_tensor({var_name}, min_input, max_input);")
        else:  # non-contiguous
            # Create a bigger tensor, then create view
            big_dims = dims.copy()
            dim_to_change = continuity.non_contiguous_dim
            op = continuity.big_tensor_op
            
            if dim_to_change is not None and op is not None and dim_to_change < len(big_dims):
                 big_dims[dim_to_change] = f"({big_dims[dim_to_change]}) {op}"

            big_shape_expr = self.generate_shape_init(shape_type, big_dims)
            
            code.append(f"// Create non-contiguous tensor (on dimension {dim_to_change})")
            code.append(f"alignas(32) auto big_tensor = ntt::make_tensor<{datatype.cpp_type}>({big_shape_expr});")
            code.append(f"NttTest::init_tensor(big_tensor, min_input, max_input);")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view_from_address<{datatype.cpp_type}>(")
            code.append(f"    big_tensor.elements().data(),")
            code.append(f"    {shape_expr},")
            code.append(f"    big_tensor.strides());")
        
        return code
    
    def generate_pack_axes_str(self, axes):
        if len(axes) == 1:
            return f"ntt::fixed_shape_v<{axes[0]}>"
        else:
            return f"ntt::fixed_shape_v<{', '.join(map(str, axes))}>"
    
    def generate_ort_reference(self, input_dims, input_dim_names, pack_axes):
        code = []
        ndim = len(input_dims)
        
        # Calculate reshaped dimensions (for code string generation)
        reshape_dims_str = []
        dim_idx = 0
        for i in range(ndim):
            if i in pack_axes:
                axis_idx = pack_axes.index(i)
                # Use string expressions instead of calculated results
                reshape_dims_str.append(f"(int64_t)({input_dim_names[i]} / P)")
                reshape_dims_str.append(f"(int64_t)P")
            else:
                reshape_dims_str.append(f"(int64_t){input_dim_names[i]}")
        
        # Generate reshape code
        code.append("// ORT reference implementation")
        code.append("auto ort_input = NttTest::ntt2ort(ntt_input);")
        code.append(f"int64_t reshape_data[] = {{{', '.join(reshape_dims_str)}}};")
        code.append("int64_t reshape_shape[] = {std::size(reshape_data)};")
        code.append("auto ort_type = NttTest::primitive_type2ort_type<int64_t>();")
        code.append("auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,")
        code.append("                         reshape_shape, std::size(reshape_shape));")
        code.append("auto reshaped_tensor = ortki_Reshape(ort_input, shape_tensor, 0);")
        
        # Generate transpose permutation
        if len(pack_axes) > 0:
            # Calculate permutation
            perm = []
            packed_dims = []
            j = 0
            for i in range(ndim):
                if i in pack_axes:
                    perm.append(j)
                    packed_dims.append(j + 1)
                    j += 2
                else:
                    perm.append(j)
                    j += 1
            perm.extend(packed_dims)
            
            code.append("")
            code.append(f"int64_t perms[] = {{{', '.join(map(str, perm))}}};")
            code.append("auto ort_output = ortki_Transpose(reshaped_tensor, perms, std::size(perms));")
        else:
            code.append("auto ort_output = reshaped_tensor;")
        
        return code
    
    def generate_test_prologue(self, datatype, test_name, P, dim_names, dims, pack_axes):
        """generate test function header, constant P and dimension constants"""
        code = [f"TEST(PackTest_{datatype.name_suffix}, {test_name}) {{", f"    constexpr size_t P = {P};"]
        
        # define dimension constants
        for i, (name, size) in enumerate(zip(dim_names, dims)):
            if i in pack_axes:
                axis_idx = pack_axes.index(i)
                code.append(f"    constexpr size_t {name}_coefficient = {size};")
                code.append(f"    constexpr size_t {name} = {name}_coefficient * P;")
            else:
                code.append(f"    constexpr size_t {name} = {size};")
        
        code.extend([f"    {datatype.cpp_type} min_input = {datatype.min_val};", 
                     f"    {datatype.cpp_type} max_input = {datatype.max_val};", ""])
        return code

    def generate_output_tensor_code(self, datatype, shape_type, dim_names, pack_axes, vector_dim):
        output_dims = []
        for i, name in enumerate(dim_names):
            if i in pack_axes:
                output_dims.append(f"{name} / P")
            else:
                output_dims.append(name)
        
        if vector_dim == 1:
            vector_type = f"ntt::vector<{datatype.cpp_type}, P>"
        else:
            vector_type = f"ntt::vector<{datatype.cpp_type}, {', '.join(['P'] * len(pack_axes))}>"
            
        output_shape_expr = self.generate_shape_init(shape_type, output_dims)
        
        code = [
            f"// Create output tensor",
            f"alignas(32) auto ntt_output1 = ntt::make_tensor<{vector_type}>({output_shape_expr});",
            ""
        ]
        return code, vector_type, output_shape_expr

    def generate_pack_call_code(self, pack_axes):
        pack_axes_str = self.generate_pack_axes_str(pack_axes)
        return [
            "// Execute pack operation",
            f"ntt::pack(ntt_input, ntt_output1, {pack_axes_str});",
            ""
        ]

    def generate_reference_and_comparison_code(self, datatype, continuity, dims, dim_names, pack_axes, shape_type, vector_type, output_shape_expr, is_fp8):
        code = []
        input_dims_expr = [f"{name}" for name in dim_names]

        ort_input_tensor = "ntt_input"
        # For non-contiguous tensor, need to copy to contiguous tensor first
        if not continuity.is_contiguous:
            if is_fp8:
                # for fp8, ntt_input_uint8 is already contiguous, created by cast
                ort_input_tensor = "ntt_input_uint8"
            else:
                code.append("    // Copy to contiguous tensor for ORT reference")
                code.append(f"    alignas(32) auto continuous_input = ntt::make_tensor<{datatype.cpp_type}>({self.generate_shape_init(shape_type, input_dims_expr)});")
                
                # generate nested loops to copy data
                code.append("    ")
                for i, name in enumerate(dim_names):
                    code.append(f"    {'    ' * i}for (size_t {name.lower()} = 0; {name.lower()} < {name}; {name.lower()}++) {{")
                
                indices = [f"{name.lower()}" for name in dim_names]
                code.append(f"    {'    ' * len(dim_names)}continuous_input({', '.join(indices)}) = ntt_input({', '.join(indices)});")
                
                for i in range(len(dim_names)-1, -1, -1):
                    code.append(f"    {'    ' * i}}}")
                code.append("")
                ort_input_tensor = "continuous_input"
        elif is_fp8: # contiguous fp8 case
            ort_input_tensor = "ntt_input_uint8"

        ort_ref = self.generate_ort_reference(dims, dim_names, pack_axes)
        # The first line of ort_ref is "// ORT reference implementation"
        # The second line is "auto ort_input = NttTest::ntt2ort(ntt_input);"
        # We modify this line.
        ort_ref[1] = f"    auto ort_input = NttTest::ntt2ort({ort_input_tensor});"
        
        code.extend([f"    {line}" for line in ort_ref])
        code.append("")
        
        # compare results
        code.append("    // Compare results")
        if is_fp8:
            vector_type_uint8 = vector_type.replace(datatype.cpp_type, 'uint8_t')
            code.append(f"    alignas(32) auto ntt_output2_uint8 = ntt::make_tensor<{vector_type_uint8}>({output_shape_expr});")
            code.append("    NttTest::ort2ntt(ort_output, ntt_output2_uint8);")
            code.append("    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_output2_uint8));")
        else:
            code.append(f"    alignas(32) auto ntt_output2 = ntt::make_tensor<{vector_type}>({output_shape_expr});")
            code.append("    NttTest::ort2ntt(ort_output, ntt_output2);")
            code.append("    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_output2));")
        code.append("}")
        code.append("")
        
        return code

# shape_type: fixed/dynamic
# vector_dim: 1/2
# continuity: is_contiguous, non_contiguous_dim, big_tensor_op
# pack_axes: list of axes to pack
# ndim: dimension of the tensor
    def generate_test_case(self, datatype, shape_type, vector_dim, continuity, pack_axes, ndim):
        # 1. initialize dimension and other basic variables
        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"
        if ndim == 3:
            dims, dim_names = [1, 77, 3], ['C', 'H', 'W']
        elif ndim == 4:
            dims, dim_names = [2, 8, 4, 4], ['N', 'C', 'H', 'W']
        else:
            dims, dim_names = [2, 8, 4, 4, 2], ['N', 'C', 'H', 'W', 'D']
        
        test_name = self.generate_test_name(datatype, shape_type, vector_dim, continuity, "_".join(map(str, pack_axes)), ndim)
        
        is_fp8 = 'float_e' in datatype.cpp_type

        # 2. call helper functions to generate code
        code = []
        
        # 2.1 generate test function header and constants
        code.extend(self.generate_test_prologue(datatype, test_name, P, dim_names, dims, pack_axes))
        
        # 2.2 generate input tensor initialization code
        input_dims_expr = [f"{name}" for name in dim_names]
        tensor_init_code = self.generate_tensor_init(datatype, shape_type, input_dims_expr, continuity, "ntt_input")
        code.extend([f"    {line}" for line in tensor_init_code])
        
        if is_fp8:
            input_shape_expr = self.generate_shape_init(shape_type, input_dims_expr)
            code.append(f"    auto ntt_input_uint8 = ntt::make_tensor<uint8_t>({input_shape_expr});")
            code.append(f"    NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);")

        code.append("")
        
        # 2.3 generate output tensor initialization code
        output_tensor_code, vector_type, output_shape_expr = self.generate_output_tensor_code(datatype, shape_type, dim_names, pack_axes, vector_dim)
        code.extend([f"    {line}" for line in output_tensor_code])

        # 2.4 generate pack operation call code
        pack_call_code = self.generate_pack_call_code(pack_axes)
        code.extend([f"    {line}" for line in pack_call_code])

        if is_fp8:
            vector_type_uint8 = vector_type.replace(datatype.cpp_type, 'uint8_t')
            code.append(f"    auto ntt_output1_uint8 = ntt::make_tensor<{vector_type_uint8}>({output_shape_expr});")
            code.append(f"    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);")
            code.append("")

        # 2.5 generate reference implementation and result comparison code
        ref_and_comp_code = self.generate_reference_and_comparison_code(datatype, continuity, dims, dim_names, pack_axes, shape_type, vector_type, output_shape_expr, is_fp8)

        code.extend(ref_and_comp_code)

        return "\n".join(code)

    def generate_all_tests_for_type(self, datatype):
        """Generate all test combinations for a given datatype
        1. rank 3, 4, 5
        2. fixed/dynamic
        3. 1D/2D vector
        4. contiguous/non-contiguous
        4.1 For dimensions 3, 5, test simple non-contiguous cases (simple_continuities)
        4.2 For dimension 4, test more complex non-contiguous cases (full_continuities)
        """
        """Uncovered test scope:
        1. Cases where packed dimensions are not multiples of P, requiring padding
        """
        shape_types = ["fixed", "dynamic"]
        vector_dims = [1, 2]
        continuities = ["contiguous", "non_contiguous"]
        
        # Define pack axis options for different dimensions
        pack_axes_options = {
            3: [[2], [1], [0], [0, 1], [1, 2]],  
            4: [[3], [2], [1], [0], [0, 1], [1, 2], [2, 3]],  
            5: [[4], [3], [2], [1], [0], [0, 1], [1, 2], [2, 3], [3, 4]]  
        }

        # Full continuity test combinations, mainly for 4D
        full_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="+7"),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="+7"),
        ]

        # Simplified continuity test combinations, for non-4D
        simple_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"), # Choose a representative non-contiguous case
        ]
        
        code = []
        
        # Generate file header
        code.append(self.generate_header())
        
        # Generate test cases
        for ndim in [3, 4, 5]:
            # Select continuity test strategy based on dimension
            current_continuities = full_continuities if ndim == 4 else simple_continuities

            for shape_type, vector_dim, continuity in itertools.product(shape_types, vector_dims, current_continuities):
                for pack_axes in pack_axes_options[ndim]:
                    # Skip unreasonable combinations
                    if vector_dim == 2 and len(pack_axes) < 2:
                        continue
                    if vector_dim == 1 and len(pack_axes) > 1:
                        continue
                    
                    test_code = self.generate_test_case(datatype, shape_type, vector_dim, continuity, pack_axes, ndim)
                    code.append(test_code)       
        # Generate main function
        code.append(self.generate_footer())
        
        return "\n".join(code)
    
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


if __name__ == "__main__":
    generator = PackTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    generated_filenames = [] # collect all generated file names

    for datatype in ALL_DATATYPES:
        test_code = generator.generate_all_tests_for_type(datatype)
        filename = f"test_ntt_pack_generated_{datatype.name_suffix}.cpp"
        output_filepath = os.path.join(script_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename) 
    
    generate_cmake_list(script_directory, generated_filenames)