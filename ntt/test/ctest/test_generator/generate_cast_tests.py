#!/usr/bin/env python3
"""
Generate test cases for NTT cast operations
Covering the following cases:
1. Input/Output type combinations: all 15 * 14 type pairs
2. Shape types: fixed/dynamic
3. Vector dimensions: scalar/1D/2D
4. Tensor continuity: contiguous/non-contiguous
5. Tensor dimensions: 3D/4D
"""

import itertools
from typing import List, Tuple
from test_generator_base import *
import os

class CastTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        
        # Element type sizes in bytes
        self.element_type_lengths = {
            'uint8_t': 1, 'int8_t': 1, 'bool': 1,
            'uint16_t': 2, 'int16_t': 2, 'half': 2, 'bfloat16': 2,
            'uint32_t': 4, 'int32_t': 4, 'float': 4,
            'uint64_t': 8, 'int64_t': 8, 'double': 8,
            'float_e4m3_t': 1, 'float_e5m2_t': 1
        }

        
    def generate_test_name(self, from_type, to_type, shape_type, vector_rank, continuity: Continuity, ndim, repackedAxes=None):
        parts = []
        parts.append(f"from_{from_type.name_suffix}_to_{to_type.name_suffix}")
        parts.append(shape_type)
        
        if vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{vector_rank}D_vector")
        
        if continuity.is_contiguous:
            parts.append("contiguous")
        else:
            op_str = "mul2" if continuity.big_tensor_op == "*2" else "add5"
            parts.append(f"non_contiguous_dim{continuity.non_contiguous_dim}_{op_str}")

        parts.append(f"{ndim}D")
        
        if repackedAxes is not None:
            parts.append(f"repack_axis_{repackedAxes[0]}")
        
        return "_".join(parts)

    def generate_ort_output(self, to_type):
        """Generate ORT reference implementation for cast operation"""
        ort_type = self.ort_datatype_map.get(to_type.cpp_type, 'DataType_FLOAT')
        return [
            "// ORT reference implementation",
            f"auto ort_output = ortki_Cast(ort_input, 1, {ort_type});",
            ""
        ]

    def generate_ntt_ops(self, repackedAxes=None):
        """Generate NTT cast operation code"""
        if repackedAxes is not None:
            axes_str = f"ntt::fixed_shape_v<{repackedAxes[0]}>"
            return [
                "// Execute cast operation",
                f"ntt::cast(ntt_input, ntt_output1, {axes_str});",
                ""
            ]
        else:
            return [
                "// Execute cast operation",
                "ntt::cast(ntt_input, ntt_output1);",
                ""
            ]


    def generate_ntt_output_to_test(self, from_type, to_type, shape_type, dims_spec, continuity, vector_rank, P, repackedAxes=None):
        """Generate the NTT output to be tested"""
        code = []

        cast_min_value, cast_max_value = clamp_value_strings(from_type, to_type) 
        cast_data_type = from_type._replace(min_val=cast_min_value, max_val=cast_max_value)
        # 1. NTT input creation
        code.extend(self.generate_tensor_init(
            datatype=cast_data_type,
            shape_type=shape_type,
            dim_spec=dims_spec,
            continuity=continuity,
            vector_rank=vector_rank,
            P=P,
            var_name="ntt_input",
            name_suffix=""))

        # 2. NTT output tensor creation
        from_ele_len = self.element_type_lengths.get(from_type.cpp_type, 4)
        to_ele_len = self.element_type_lengths.get(to_type.cpp_type, 4)
        if(from_type.cpp_type == "bool"):
            from_ele_len = to_ele_len
        if(to_type.cpp_type == "bool"):
            to_ele_len = from_ele_len

        # Calculate output P and adjust dimensions for 1D vector with different element type length
        if vector_rank == 1 and repackedAxes is not None and from_ele_len != to_ele_len:
            # Adjust the P of output tensor: P = P / (output_ele_len / input_ele_len)
            scale_factor = to_ele_len // from_ele_len if to_ele_len > from_ele_len else from_ele_len // to_ele_len
            if to_ele_len > from_ele_len:
                output_P = f"P / {scale_factor}"
            else:
                output_P = f"P * {scale_factor}"
            
            # Adjust the axes of output tensor: axes[repackedAxes[0]] = axes[repackedAxes[0]] * (output_ele_len / input_ele_len)
            adjusted_dims_spec = [str(dim) for dim in dims_spec]
            repack_axis = repackedAxes[0]
            if to_ele_len > from_ele_len:
                adjusted_dims_spec[repack_axis] = f"{dims_spec[repack_axis]} * {scale_factor}"
            else:
                adjusted_dims_spec[repack_axis] = f"{dims_spec[repack_axis]} / {scale_factor}"
            output_element_type = self.get_element_cpp_type(to_type.cpp_type, vector_rank, output_P)
            output_shape_expr = self.generate_shape_init(shape_type, adjusted_dims_spec)
        else:
            output_element_type = self.get_element_cpp_type(to_type.cpp_type, vector_rank, P)
            output_shape_expr = self.generate_shape_init(shape_type, [str(dim) for dim in dims_spec])

        code.append(f"// Create output tensor")
        code.append(f"auto ntt_output1 = ntt::make_tensor<{output_element_type}>({output_shape_expr});")
        code.append("")

        # 3. NTT operation (cast)
        cast_call_code = self.generate_ntt_ops(repackedAxes)

        op_code = self.generate_ntt_operation_section(cast_call_code)
        code.extend(op_code)

        return code, output_shape_expr, output_element_type

    # for fp8, golden is derived from the apply operation to cast tensor elementwisely.
    def generate_ntt_cast_golden_output_fp8(self, from_type, to_type, shape_type, dims_spec, continuity, P, vector_rank):
        code = []
        from_ele_len = self.element_type_lengths.get(from_type.cpp_type, 4)
        to_ele_len = self.element_type_lengths.get(to_type.cpp_type, 4)
        scale = from_ele_len // to_ele_len if from_ele_len > to_ele_len else to_ele_len // from_ele_len
        tensor_element_type = self.get_element_cpp_type(from_type.cpp_type, vector_rank, P)


        output_element_type = self.get_element_cpp_type(to_type.cpp_type, vector_rank, P)

        # 1. copy to contiguous tensor of scalar or vector
        if not continuity.is_contiguous:
            copy_code, continuous_input_var_name = self.generate_copy_to_contiguous_code(tensor_element_type, shape_type, dims_spec)
            code.extend(copy_code)
        else:
            continuous_input_var_name = "ntt_input"

        unpack_axes = [len(dims_spec)-1] if vector_rank == 1 else [len(dims_spec)-2, len(dims_spec)-1]
        # 2. unpack to scalar tensor
        if 'vector' in tensor_element_type:
            # 2x4 vector<int32, 8> to 2x(4/2) vector<int16, 8*2>
            out_vec_dims_spec = dims_spec.copy()
            out_vec_dims_spec[-1]  = ((dims_spec[-1] // scale) if from_ele_len > to_ele_len 
                                    else (dims_spec[-1] * scale))
            

            output_element_type = self.get_element_cpp_type(
                    to_type.cpp_type, vector_rank, 
                    (f"{P} * {scale}" if from_ele_len > to_ele_len else f"{P} / {scale}") )

            unpacked_dims = self.get_unpacked_dims(dims_spec, unpack_axes)
            code.append(f"auto ntt_scalar_input = ntt::make_tensor<{from_type.cpp_type}>({self.generate_shape_init(shape_type, unpacked_dims)});")
            code.append(f"ntt::unpack({continuous_input_var_name}, ntt_scalar_input, {self.generate_pack_axes_str(unpack_axes)});")
        else:
            code.append(f"auto ntt_scalar_input = {continuous_input_var_name};")
        #3. generate golden output
        code.append(f"auto ntt_golden_scalar = ntt::make_tensor<{to_type.cpp_type}>(ntt_scalar_input.shape());")
        code.append(
            f"ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){{\n"
            f"      (ntt_golden_scalar)(index) = static_cast<{to_type.cpp_type}>(ntt_scalar_input(index));\n"
            f"    }});"
        )

        # 4. generate under test scalar output 
        if "vector" in tensor_element_type:
            code.append(f"auto ntt_golden_vector = ntt::make_tensor<{output_element_type}>({self.generate_shape_init(shape_type, out_vec_dims_spec)});")
            code.append(f"ntt::pack(ntt_golden_scalar, ntt_golden_vector, {self.generate_pack_axes_str(unpack_axes)});")
            code.append(f"auto& ntt_golden = ntt_golden_vector;")
        else:
            code.append(f"auto& ntt_golden = ntt_golden_scalar;")


        return code


    def generate_ort_golden_output(self, from_type, to_type, shape_type, dims_spec, continuity, P,  vector_rank, deal_fp8, repackedAxes=None):
        """Generate golden output using ORT or lambda-based reference"""
        code = []
        is_fp8_cast = 'float_e' in from_type.cpp_type or 'float_e' in to_type.cpp_type

        if not is_fp8_cast:
            # Generate ORT input section using prepare_contiguous_input
            continuity_var_name, copy_code = self.prepare_contiguous_input(
                "ntt_input", from_type, vector_rank, P,
                shape_type, dims_spec, continuity
            )
            code.extend(copy_code)
            ort_input_var_name = continuity_var_name
            
            # Add logic for 1D vector with different element type lengths
            from_ele_len = self.element_type_lengths.get(from_type.cpp_type, 4)
            to_ele_len = self.element_type_lengths.get(to_type.cpp_type, 4)
            scale = from_ele_len // to_ele_len if from_ele_len > to_ele_len else to_ele_len // from_ele_len
            if("bool" == from_type.cpp_type or "bool" == to_type.cpp_type):
                scale = 1

            if vector_rank == 1 and repackedAxes is not None and from_ele_len != to_ele_len:
                # Calculate scale factor
                input_rank = len(dims_spec)
                packed_axis = repackedAxes[0]
                
                ort_dims_spec = [str(dim) for dim in dims_spec]
                if from_ele_len > to_ele_len:
                    ort_dims_spec[packed_axis] = f"{ort_dims_spec[packed_axis]} / {scale}"
                    ort_dims_spec.insert(packed_axis + 1, str(scale))
                    ort_dims_spec.append("P")
                    
                    # Create perms_data: move second to last element after packed_axis
                    perms_data = list(range(input_rank + 2))
                    # Move element at index input_rank-1 (the second to last element) to position packed_axis + 1
                    element = perms_data.pop(input_rank ) # input_rank + 1 - 1
                    perms_data.insert(packed_axis + 1, element)
                else:
                    # scale < 1 case (from_ele_len < to_ele_len)
                    ort_dims_spec.append(str(scale))
                    ort_dims_spec.append(f"P / {scale}")
                    
                    # Create perms_data: move second to last element after packed_axis
                    perms_data = list(range(input_rank + 2))
                    # Move element at index input_rank (the second to last element) to position packed_axis + 1
                    element = perms_data.pop(input_rank ) # input_rank + 1 - 1
                    perms_data.insert(packed_axis + 1, element)
                
                
                # Generate reshape and transpose code
                code.append("// Reshape and transpose for 1D vector cast")
                code.append(f"int64_t reshape_data[] = {{{', '.join(ort_dims_spec)}}};")
                code.append(f"int64_t reshape_shape[] = {{std::size(reshape_data)}};")
                code.append("auto ort_type = NttTest::primitive_type2ort_type<int64_t>();")
                code.append("auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,")
                code.append("                         reshape_shape, std::size(reshape_shape));")
                code.append(f"auto ort_input = NttTest::ntt2ort({ort_input_var_name});")
                code.append(f"auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);")
                code.append("")
                code.append(f"int64_t perms_data[] = {{{', '.join(map(str, perms_data))}}};")
                code.append("auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));")
                code.append("")
                
                # Use the transposed tensor as input for cast
                ort_type = self.ort_datatype_map.get(to_type.cpp_type, 'DataType_FLOAT')
                code.append(f"auto ort_output = ortki_Cast(ort_cast_input, 1, {ort_type});")
            else:
                # Use standard ORT output
                code.append(f"auto ort_input = NttTest::ntt2ort({ort_input_var_name});")
                ort_kernel_lines = self.generate_ort_output(to_type)
                code.extend(ort_kernel_lines)
        else:
            # Use lambda-based reference
            code.extend(self.generate_ntt_cast_golden_output_fp8(from_type, to_type, shape_type, dims_spec, continuity, P, vector_rank))
            
        return code
    


    def generate_test_case(self, from_type, to_type, shape_type, vector_rank, continuity, ndim, repackedAxes=None):
        """Generate a single test case"""
        # 1. Initialize dimensions and other basic variables
        is_from_fp8 = 'float_e' in from_type.cpp_type
        is_to_fp8 = 'float_e' in to_type.cpp_type
        deal_fp8 = 1 if (is_from_fp8 or is_to_fp8) else 0
        is_fp8_cast = is_from_fp8 or is_to_fp8

        vector_element = from_type.cpp_type if from_type.cpp_type != "bool" else to_type.cpp_type

        P = f"NTT_VLEN / (sizeof({vector_element}) * 8)"
        if ndim == 3:
            dims_spec = [8, 80, 8]
        elif ndim == 4:
            dims_spec= [8, 16, 8, 8]
        else:
            dims_spec= [2, 8, 4, 4, 4]


        test_name = self.generate_test_name(from_type, to_type, shape_type, vector_rank, continuity, ndim, repackedAxes)
        
        code: List[str] = []


        # 1. Test header and constants
        code.extend(self.generate_function_name("CastTest", from_type, test_name))
        P_would_be_used = True if vector_rank > 0 else False

        if(P_would_be_used):
            code.extend(self.generate_P_constants(P))
        # 2. Generate output to test in NTT format
        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(
            from_type, to_type, shape_type, dims_spec, continuity, vector_rank, "P", repackedAxes)
        code.extend([f"    {line}" for line in ntt_output_code])

        # 3. Generate golden output in ORT format, or in ntt format for fp8 cast
        golden_output_code = self.generate_ort_golden_output(
            from_type, to_type, shape_type, dims_spec, continuity, P,  vector_rank, deal_fp8, repackedAxes)
    
        code.extend([f"    {line}" for line in golden_output_code])

        # 4. Compare outputs
        if is_fp8_cast:
            # Direct comparison for FP8 cast
            code.extend([
                "    // Compare results",
                "    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));",
                "}"
            ])
        else:
            # ORT-based comparison
            compare_code = self.generate_ort_back2ntt_and_compare_section(
                to_type,
                output_element_type,
                output_shape_expr,
                deal_fp8,
                ntt_output_var_name="ntt_output1",
                ort_output_var_name="ort_output")
            code.extend([f"    {line}" for line in compare_code])

        return "\n".join(code)

    def generate_all_tests_for_from_type(self, from_type):
        """Generate all test combinations for a given input datatype"""
        shape_types = ["fixed", "dynamic"]
        vector_ranks = [0, 1, 2]  # scalar, 1D vector, 2D vector
        
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
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
        ]
        
        code = []
        
        # Generate file header
        code.append(self.generate_header())
        
        # Generate test cases for all target types (except the same type)
        for to_type in ALL_DATATYPES:
            if from_type.cpp_type == to_type.cpp_type:
                continue  # Skip same type cast
            
            # Generate test cases for different dimensions
            for ndim in [3, 4]:
                # Select continuity test strategy based on dimension
                current_continuities = full_continuities if ndim == 3 else simple_continuities

                for shape_type, vector_rank, continuity in itertools.product(shape_types, vector_ranks, current_continuities):
                    # Skip unreasonable combinations
                    if vector_rank > ndim:  # Can't have more vector dimensions than tensor dimensions
                        continue
                    
                    # Determine repackedAxes choices based on vector_rank and element type lengths
                    from_ele_len = self.element_type_lengths.get(from_type.cpp_type, 4)
                    to_ele_len = self.element_type_lengths.get(to_type.cpp_type, 4)
                    
                    repackedAxes_choices = []
                    
                    if vector_rank == 0:  # scalar
                        # No filter, but repackedAxes should not be set
                        repackedAxes_choices = [None]
                    elif vector_rank == 1:  # 1D vector
                        if from_ele_len == to_ele_len:
                            # For equal element type length, repackedAxes should not be set
                            repackedAxes_choices = [None]
                        else:
                            # For different element type length, add repackedAxes
                            # Only test with repackedAxes when element type lengths differ
                            repackedAxes_choices = [[ndim-1]]  # Add repack on last axis
                    elif vector_rank == 2:  # 2D vector
                        # Only element type length equal case should be tested
                        if from_ele_len == to_ele_len:
                            repackedAxes_choices = [None]
                        else:
                            continue  # Skip different element type length for 2D vector
                    # if "float_e" in to_type.cpp_type :
                    #     continue
                    
                    for repackedAxes in repackedAxes_choices:
                        test_code = self.generate_test_case(from_type, to_type, shape_type, vector_rank, continuity, ndim, repackedAxes)
                        code.append(test_code)
                    
        # Generate main function
        code.append(self.generate_footer())
        
        return "\n".join(code)


if __name__ == "__main__":
    generator = CastTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (ctest) and then the generated subdirectory
    ctest_directory = os.path.dirname(script_directory)
    generated_directory = os.path.join(ctest_directory, "generated")
    
    # Ensure generated directory exists
    os.makedirs(generated_directory, exist_ok=True)
    
    generated_filenames = []  # collect all generated file names

    for from_type in ALL_DATATYPES:
        test_code = generator.generate_all_tests_for_from_type(from_type)
        filename = f"test_ntt_cast_from_{from_type.name_suffix.lower()}_generated.cpp"
        output_filepath = os.path.join(generated_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)
    
    # Generate cmake list file in the generated directory
    generate_cmake_list(generated_directory, generated_filenames, "generated_cast_tests.cmake", "GENERATED_CAST_TEST_SOURCES") 