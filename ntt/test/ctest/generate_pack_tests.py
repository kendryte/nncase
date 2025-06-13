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


# is_contiguous: bool 
# non_contiguous_dim: int or None 
# big_tensor_op: str or None -  How to build the big tensor at given non_contiguous_dim
Continuity = namedtuple('Continuity', ['is_contiguous', 'non_contiguous_dim', 'big_tensor_op'])

class PackTestGenerator:
    def __init__(self):
        self.test_cases = []
        
    def generate_test_name(self, shape_type, vector_dim, continuity: Continuity, pack_axis_str, ndim):
        parts = []
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
    
    def generate_tensor_init(self, shape_type, dims, continuity, var_name):
        code = []
        shape_expr = self.generate_shape_init(shape_type, dims)
        
        if continuity.is_contiguous:
            code.append(f"alignas(32) auto {var_name} = ntt::make_tensor<float>({shape_expr});")
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
            code.append(f"alignas(32) auto big_tensor = ntt::make_tensor<float>({big_shape_expr});")
            code.append(f"NttTest::init_tensor(big_tensor, min_input, max_input);")
            code.append(f"")
            code.append(f"auto {var_name} = ntt::make_tensor_view_from_address<float>(")
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
    
    def generate_test_prologue(self, test_name, P, dim_names, dims, pack_axes):
        """generate test function header, constant P and dimension constants"""
        code = [f"TEST(PackTestFloat, {test_name}) {{", f"    constexpr size_t P = {P};"]
        
        # define dimension constants
        for i, (name, size) in enumerate(zip(dim_names, dims)):
            if i in pack_axes:
                axis_idx = pack_axes.index(i)
                code.append(f"    constexpr size_t {name}_coefficient = {size};")
                code.append(f"    constexpr size_t {name} = {name}_coefficient * P;")
            else:
                code.append(f"    constexpr size_t {name} = {size};")
        
        code.extend(["    float min_input = -10.0f;", "    float max_input = 10.0f;", ""])
        return code

    def generate_output_tensor_code(self, shape_type, dim_names, pack_axes, vector_dim):
        output_dims = []
        for i, name in enumerate(dim_names):
            if i in pack_axes:
                output_dims.append(f"{name} / P")
            else:
                output_dims.append(name)
        
        if vector_dim == 1:
            vector_type = f"ntt::vector<float, P>"
        else:
            vector_type = f"ntt::vector<float, {', '.join(['P'] * len(pack_axes))}>"
            
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

    def generate_reference_and_comparison_code(self, continuity, dims, dim_names, pack_axes, shape_type, vector_type, output_shape_expr):
        code = []
        input_dims_expr = [f"{name}" for name in dim_names]

        # For non-contiguous tensor, need to copy to contiguous tensor first
        if not continuity.is_contiguous:
            code.append("    // Copy to contiguous tensor for ORT reference")
            code.append(f"    alignas(32) auto continuous_input = ntt::make_tensor<float>({self.generate_shape_init(shape_type, input_dims_expr)});")
            
            # generate nested loops to copy data
            code.append("    ")
            for i, name in enumerate(dim_names):
                code.append(f"    {'    ' * i}for (size_t {name.lower()} = 0; {name.lower()} < {name}; {name.lower()}++) {{")
            
            indices = [f"{name.lower()}" for name in dim_names]
            code.append(f"    {'    ' * len(dim_names)}continuous_input({', '.join(indices)}) = ntt_input({', '.join(indices)});")
            
            for i in range(len(dim_names)-1, -1, -1):
                code.append(f"    {'    ' * i}}}")
            code.append("")
            
            # let ORT use the copied contiguous tensor
            ort_ref = self.generate_ort_reference(dims, dim_names, pack_axes)
            ort_ref[1] = "    auto ort_input = NttTest::ntt2ort(continuous_input);"
        else:
            ort_ref = self.generate_ort_reference(dims, dim_names, pack_axes)
        
        code.extend([f"    {line}" for line in ort_ref])
        code.append("")
        
        # compare results
        code.append("    // Compare results")
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
    def generate_test_case(self, shape_type, vector_dim, continuity, pack_axes, ndim):
        # 1. initialize dimension and other basic variables
        P = "NTT_VLEN / (sizeof(float) * 8)"
        if ndim == 3:
            dims, dim_names = [1, 77, 3], ['C', 'H', 'W']
        elif ndim == 4:
            dims, dim_names = [2, 8, 4, 4], ['N', 'C', 'H', 'W']
        else:
            dims, dim_names = [2, 8, 4, 4, 2], ['N', 'C', 'H', 'W', 'D']
        
        test_name = self.generate_test_name(shape_type, vector_dim, continuity, "_".join(map(str, pack_axes)), ndim)
        
        # 2. call helper functions to generate code
        code = []
        
        # 2.1 generate test function header and constants
        code.extend(self.generate_test_prologue(test_name, P, dim_names, dims, pack_axes))
        
        # 2.2 generate input tensor initialization code
        input_dims_expr = [f"{name}" for name in dim_names]
        tensor_init_code = self.generate_tensor_init(shape_type, input_dims_expr, continuity, "ntt_input")
        code.extend([f"    {line}" for line in tensor_init_code])
        code.append("")
        
        # 2.3 generate output tensor initialization code
        output_tensor_code, vector_type, output_shape_expr = self.generate_output_tensor_code(shape_type, dim_names, pack_axes, vector_dim)
        code.extend([f"    {line}" for line in output_tensor_code])

        # 2.4 generate pack operation call code
        pack_call_code = self.generate_pack_call_code(pack_axes)
        code.extend([f"    {line}" for line in pack_call_code])
        
        # 2.5 generate reference implementation and result comparison code
        ref_and_comp_code = self.generate_reference_and_comparison_code(continuity, dims, dim_names, pack_axes, shape_type, vector_type, output_shape_expr)

        code.extend(ref_and_comp_code)

        return "\n".join(code)

    def generate_all_tests(self):
        """Generate all test combinations
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
                    
                    test_code = self.generate_test_case(shape_type, vector_dim, continuity, pack_axes, ndim)
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

if __name__ == "__main__":
    generator = PackTestGenerator()
    test_code = generator.generate_all_tests()
    
    # Write to file
    with open("test_ntt_pack_generated.cpp", "w") as f:
        f.write(test_code)
    
    print("Test file generated: test_ntt_pack_generated.cpp") 