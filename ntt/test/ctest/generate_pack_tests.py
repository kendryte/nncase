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
from test_generator_base import *
import os

class PackTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        
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
        code.append("// ORT reference implementation (kernel part)")
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
    
    def generate_ntt_ops(self, pack_axes):
        pack_axes_str = self.generate_pack_axes_str(pack_axes)
        return [
            "// Execute pack operation",
            f"ntt::pack(ntt_input, ntt_output1, {pack_axes_str});",
            ""
        ]

    def generate_ntt_output_to_test(self, datatype, shape_type, dim_names, continuity, vector_dim, P, pack_axes, deal_fp8):
        """
        Generates the NTT output to be tested.
        This includes:
        1. Creating the NTT input tensor.
        2. Creating the NTT output tensor.
        3. Calling the ntt::pack operation.
        4. Handling FP8 reinterpret_cast for the output if necessary.
        """
        code = []

        # 1. NTT input creation
        code.extend(self.generate_ntt_input_section(
            datatype=datatype,
            shape_type=shape_type,
            dim_names=dim_names,
            continuity=continuity,
            vector_rank=0,  # Pack input is always scalar tensor
            P=P,
            axes_count=len(pack_axes),
            var_name="ntt_input"))

        # 2. NTT operation (pack)
        output_dims = []
        for i, name in enumerate(dim_names):
            if i in pack_axes:
                output_dims.append(f"{name} / P")
            else:
                output_dims.append(name)
        output_shape_expr = self.generate_shape_init(shape_type, output_dims)
        
        output_element_type = self._build_vector_cpp_type(
            datatype.cpp_type, vector_dim, 'P', len(pack_axes))

        pack_call_code = self.generate_ntt_ops(pack_axes)

        op_code = self.generate_ntt_output_and_op_section(
            datatype=datatype,
            output_shape_expr=output_shape_expr,
            deal_fp8=deal_fp8,
            ntt_op_call_lines=pack_call_code,
            output_element_type=output_element_type
        )
        code.extend(op_code)
        
        return code, output_shape_expr, output_element_type

    def generate_ntt_golden_output(self, datatype, shape_type, dims, dim_names, continuity, P, pack_axes, deal_fp8):
        """
        Generates the golden output using ORT as a reference.
        This includes:
        1. Creating the ORT input.
        2. Executing the ORT reference implementation.
        """
        code = []

        # 1. ORT input section
        code.extend(self.generate_ort_input_section(
            datatype=datatype,
            shape_type=shape_type,
            dim_names=dim_names,
            continuity=continuity,
            deal_fp8=deal_fp8,
            P=P,
            vector_rank=0, # Pack input is scalar
            axes_count=len(pack_axes),
            ntt_input_var_name="ntt_input"))

        # 2. ORT kernel exec section
        ort_kernel_lines = self.generate_ort_reference(dims, dim_names, pack_axes)
        code.extend(self.generate_ort_operation_section(ort_kernel_lines))
        return code

# shape_type: fixed/dynamic
# vector_dim: 1/2
# continuity: is_contiguous, non_contiguous_dim, big_tensor_op
# pack_axes: list of axes to pack
# ndim: dimension of the tensor
    def generate_test_case(self, datatype, shape_type, vector_dim, continuity, pack_axes, ndim):
        # 1. initialize dimension and other basic variables
        is_fp8_type = 'float_e' in datatype.cpp_type
        deal_fp8 = 1 if is_fp8_type else 0

        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"
        if ndim == 3:
            dims, dim_names = [1, 77, 3], ['C', 'H', 'W']
        elif ndim == 4:
            dims, dim_names = [2, 8, 4, 4], ['N', 'C', 'H', 'W']
        else:
            dims, dim_names = [2, 8, 4, 4, 2], ['N', 'C', 'H', 'W', 'D']
        
        test_name = self.generate_test_name(datatype, shape_type, vector_dim, continuity, "_".join(map(str, pack_axes)), ndim)
        
        code: List[str] = []

        # 1. Test header and constants
        code.extend(self.generate_test_prologue("PackTest", datatype, test_name, P, dim_names, dims, pack_axes))

        # 2. Generate output to test in ntt format
        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(
            datatype, shape_type, dim_names, continuity, vector_dim, P, pack_axes, deal_fp8)
        code.extend([f"    {line}" for line in ntt_output_code])

        # 3. Generate golden output in ort format
        golden_output_code = self.generate_ntt_golden_output(
            datatype, shape_type, dims, dim_names, continuity, P, pack_axes, deal_fp8)
        code.extend([f"    {line}" for line in golden_output_code])

        # 4. Compare outputs
        compare_code = self.generate_ort_back2ntt_and_compare_section(
            datatype,
            output_element_type,
            output_shape_expr,
            deal_fp8,
            ntt_output_var_name="ntt_output1",
            ort_output_var_name="ort_output")
        code.extend([f"    {line}" for line in compare_code])

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
                    if vector_dim != len(pack_axes):
                        continue
                    
                    test_code = self.generate_test_case(datatype, shape_type, vector_dim, continuity, pack_axes, ndim)
                    code.append(test_code)       
        # Generate main function
        code.append(self.generate_footer())
        
        return "\n".join(code)
    
    


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
    
    generate_cmake_list(script_directory, generated_filenames, "generated_pack_tests.cmake", "GENERATED_PACK_TEST_SOURCES")