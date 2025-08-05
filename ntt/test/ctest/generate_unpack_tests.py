#!/usr/bin/env python3
"""
Generate test cases for NTT unpack operations
Covering the following cases:
1. Shape types: fixed/dynamic
2. Vector dimensions: 1D/2D
3. Tensor continuity: contiguous/non-contiguous
4. Unpack axes: different dimensions
"""

import itertools
import os
from typing import List
from test_generator_base import *


class UnpackTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()

    def generate_test_name(self, datatype, shape_type, vector_dim, continuity: Continuity, unpack_axis_str, ndim):
        parts = []
        parts.append(datatype.name_suffix)
        parts.append(shape_type)
        parts.append(f"{vector_dim}D_vector")

        if continuity.is_contiguous:
            parts.append("contiguous")
        else:
            op_str = "mul2" if continuity.big_tensor_op == "*2" else "add5"
            parts.append(f"non_contiguous_dim{continuity.non_contiguous_dim}_{op_str}")

        parts.append(f"unpack_axis_{unpack_axis_str}")
        parts.append(f"{ndim}D")
        return "_".join(parts)


    def generate_unpack_axes_str(self, axes):
        if len(axes) == 1:
            return f"ntt::fixed_shape_v<{axes[0]}>"
        else:
            return f"ntt::fixed_shape_v<{', '.join(map(str, axes))}>"

    def generate_ort_reference(self, input_dims, input_dim_names, unpack_axes, P):
        code = []
        ndim = len(input_dims)

        # Unpack ORT reference:
        # 1. Transpose to move the expanded vector dimensions to the correct axes
        # 2. Reshape to get the final unpackd tensor
        code.append("// ORT reference implementation (kernel part)")

        if len(unpack_axes) > 0:
            # 1. Generate transpose permutation
            perm = []
            p_map = {axis: ndim + i for i, axis in enumerate(unpack_axes)}
            for i in range(ndim):
                perm.append(i)
                if i in p_map:
                    perm.append(p_map[i])

            code.append(f"int64_t perms[] = {{{', '.join(map(str, perm))}}};")
            code.append("auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));")
            reshape_source = "transposed_tensor"
        else:
            reshape_source = "ort_input"

        # 2. Reshape to final output shape
        output_dims = []
        for i, name in enumerate(input_dim_names):
            if i in unpack_axes:
                output_dims.append(f"{name} * P")
            else:
                output_dims.append(name)

        code.append(f"int64_t reshape_data[] = {{{', '.join(output_dims)}}};")
        code.append("int64_t reshape_shape[] = {std::size(reshape_data)};")
        code.append("auto ort_type = NttTest::primitive_type2ort_type<int64_t>();")
        code.append("auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,")
        code.append("                         reshape_shape, std::size(reshape_shape));")
        code.append(f"auto ort_output = ortki_Reshape({reshape_source}, shape_tensor, 0);")

        return code


    def generate_ntt_ops(self, unpack_axes):
        unpack_axes_str = self.generate_unpack_axes_str(unpack_axes)
        return [
            "// Execute unpack operation",
            f"ntt::unpack(ntt_input, ntt_output1, {unpack_axes_str});",
            ""
        ]

    def generate_ntt_output_to_test(self, datatype, shape_type, dim_names, continuity, vector_dim, P, unpack_axes, deal_fp8):
        """
        Generates the NTT output to be tested.
        This includes:
        1. Creating the NTT input tensor.
        2. Creating the NTT output tensor.
        3. Calling the ntt::unpack operation.
        4. Handling FP8 reinterpret_cast for the output if necessary.
        """
        code = []

        # 1. NTT input creation
        code.extend(self.generate_ntt_input_section(
            datatype=datatype,
            shape_type=shape_type,
            dim_names=dim_names,
            continuity=continuity,
            vector_rank=vector_dim,
            P=P,
            axes_count=len(unpack_axes),
            var_name="ntt_input"))

        # 2. NTT operation (unpack)
        output_dims = []
        for i, name in enumerate(dim_names):
            if i in unpack_axes:
                output_dims.append(f"{name} * P")
            else:
                output_dims.append(name)
        output_shape_expr = self.generate_shape_init(shape_type, output_dims)
        
        unpack_call_code = self.generate_ntt_ops(unpack_axes)

        op_code = self.generate_ntt_output_and_op_section(
            datatype=datatype,
            output_shape_expr=output_shape_expr,
            deal_fp8=deal_fp8,
            ntt_op_call_lines=unpack_call_code
        )
        code.extend(op_code)
        
        return code, output_shape_expr

    def generate_ntt_golden_output(self, datatype, shape_type, dims, dim_names, continuity, vector_dim, P, unpack_axes, deal_fp8, output_shape_expr):
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
            vector_rank=vector_dim,
            axes_count=len(unpack_axes),
            ntt_input_var_name="ntt_input"))

        # 2. ORT kernel exec section
        ort_kernel_lines = self.generate_ort_reference(dims, dim_names, unpack_axes, P)
        code.extend(self.generate_ort_operation_section(ort_kernel_lines))
        return code

    def generate_test_case(self, datatype, shape_type, vector_dim, continuity, unpack_axes, ndim):
        is_fp8_type = 'float_e' in datatype.cpp_type
        deal_fp8 = 1 if is_fp8_type else 0

        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"
        if ndim == 3:
            dims, dim_names = [1, 77, 3], ['C', 'H', 'W']
        elif ndim == 4:
            dims, dim_names = [2, 8, 4, 4], ['N', 'C', 'H', 'W']
        else:
            dims, dim_names = [2, 8, 4, 4, 2], ['N', 'C', 'H', 'W', 'D']

        test_name = self.generate_test_name(datatype, shape_type, vector_dim, continuity, "_".join(map(str, unpack_axes)), ndim)

        code: List[str] = []

        # 1. Test header and constants
        code.extend(self.generate_test_prologue("UnpackTest", datatype, test_name, P, dim_names, dims))
        
        # Generate output to test in ntt format
        ntt_output_code, output_shape_expr = self.generate_ntt_output_to_test(datatype, shape_type, dim_names, continuity, vector_dim, P, unpack_axes, deal_fp8)
        code.extend([f"    {line}" for line in ntt_output_code])

        # Generate golden output in ort format
        golden_output_code = self.generate_ntt_golden_output(datatype, shape_type, dims, dim_names, continuity, vector_dim, P, unpack_axes, deal_fp8, output_shape_expr)
        code.extend([f"    {line}" for line in golden_output_code])

        # Compare outputs
        compare_code = self.generate_ort_back2ntt_and_compare_section(
            datatype,
            datatype.cpp_type,
            output_shape_expr,
            deal_fp8,
            ntt_output_var_name="ntt_output1",
            ort_output_var_name="ort_output")
        code.extend([f"    {line}" for line in compare_code])

        return "\n".join(code)

    def generate_all_tests_for_type(self, datatype):
        shape_types = ["fixed", "dynamic"]
        vector_dims = [1, 2]

        unpack_axes_options = {
            3: [[2], [1], [0], [0, 1], [1, 2]],
            4: [[3], [2], [1], [0], [0, 1], [1, 2], [2, 3]],
            5: [[4], [3], [2], [1], [0], [0, 1], [1, 2], [2, 3], [3, 4]]
        }

        full_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="+7"),
            Continuity(is_contiguous=False, non_contiguous_dim=2, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="+7"),
        ]

        simple_continuities = [
            Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None),
            Continuity(is_contiguous=False, non_contiguous_dim=1, big_tensor_op="*2"),
        ]

        code = []

        code.append(self.generate_header())

        for ndim in [3, 4, 5]:
            current_continuities = full_continuities if ndim == 4 else simple_continuities

            for shape_type, vector_dim, continuity in itertools.product(shape_types, vector_dims, current_continuities):
                for unpack_axes in unpack_axes_options[ndim]:
                    # The vector dimension must match the number of axes to unpack.
                    if vector_dim != len(unpack_axes) and vector_dim > 0:
                        continue
                    
                    test_code = self.generate_test_case(datatype, shape_type, vector_dim, continuity, unpack_axes, ndim)
                    code.append(test_code)

        code.append(self.generate_footer())

        return "\n".join(code)


if __name__ == "__main__":
    generator = UnpackTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))

    generated_filenames = []

    for datatype in ALL_DATATYPES:
        test_code = generator.generate_all_tests_for_type(datatype)
        filename = f"test_ntt_unpack_generated_{datatype.name_suffix}.cpp"
        output_filepath = os.path.join(script_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)

        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)

    generate_cmake_list(script_directory, generated_filenames, "generated_unpack_tests.cmake", "GENERATED_UNPACK_TEST_SOURCES")
