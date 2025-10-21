import itertools
import os
from typing import List
from test_generator_base import *



class UnaryTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        self.op_str_map_exhaustive= {
            "abs": f"auto ort_output = ortki_Abs(ort_input);", 
            "cos": f"auto ort_output = ortki_Cos(ort_input);",
            "sin": f"auto ort_output = ortki_Sin(ort_input);",
            "exp": f"auto ort_output = ortki_Exp(ort_input);"
            # "acos", "acosh", "asin", "asinh", "ceil", "copy", "cos", "cosh",
            # "exp", "erf", "floor", "log", "neg", "round", "rsqrt", "sign", "sin",
            # "sinh", "sqrt", "square", "tanh", "swish",
        }
        self.op_str_map_simplified = {

        }

        self.ort_custom_function = {

        }

        self.dims_specs_options = {
            "default": [
                [2, 3, 16, 16],
                [2, 1, 16, 7]
            ]
        }

        # ORT *unary operations* do not support these data types, need to cast to double 
        # fortunately, they could be *cast* in ort( fp8 are unfortunate)
        default_cast_types_in_ort = [
            'bool', 'int8_t', 'int16_t', 'bfloat16', 'half',
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
        ]

        self.types_need_cast_in_ort = {
            "sin": default_cast_types_in_ort + ['int32_t', 'int64_t'],
            "cos": default_cast_types_in_ort + ['int32_t', 'int64_t', 'double'],
            "default": default_cast_types_in_ort,
            "exp": default_cast_types_in_ort + ['int32_t', 'uint64_t', 'int64_t'],
        }
        self.types_need_cast_in_ntt = {
            'float_e4m3_t', 'float_e5m2_t' 
        }

        self.OPRATOR_SPECICAL_TYPES = {
            "sin": {
                "float": DataType("float", "Float32", "-1e6", "1e6", False)
            },
            "cos": {
                "float": DataType("float", "Float32", "-1e6", "1e6", False),
                "double": DataType("double", "Float64", "-1e6", "1e6", False),
            },
            "exp": {
                "float": DataType("float", "Float32", "-70.0", "70.0", False),
                "double": DataType("double", "Float64", "-700.0", "700.0", False),
                "half": DataType("half", "Float16", "half(-10.0f)", "half(10.0f)", False),
                "bfloat16": DataType("bfloat16", "Bfloat16", "-70.0_bf16", "70.0_bf16", False),
                "float_e4m3_t": DataType("float_e4m3_t", "Float8e4m3", "float_e4m3_t(-6.0f)", "float_e4m3_t(6.0f)", False),
                "uint8_t": DataType('uint8_t', 'Uint8', '0', '5', True),
                "int8_t": DataType('int8_t', 'Int8', '-4', '4', True),
                "uint16_t": DataType('uint16_t', 'Uint16', '0', '11', True),
                "int16_t": DataType('int16_t', 'Int16', '-10', '10', True),
                "uint32_t": DataType('uint32_t', 'Uint32', '0', '22', True),
                "int32_t": DataType('int32_t', 'Int32', '-21', '21', True),
                "uint64_t": DataType('uint64_t', 'Uint64', '0', '30', True),
                "int64_t": DataType('int64_t', 'Int64', '-30', '30', True)
            }
        }


        self.ulp_tolerances  = {
            "sin": {
                "default": 2
            },
            "cos":{
                "default": 2
            },
            "default": {
                "default": 1
            }
        }
    def _get_param_combinations(self, op_str):
        is_dynamic_options = [False, True]
        is_view_options = [False, True]
        vector_rank_options = [0, 1, 2]  # 0: tensor, 1: 1d vector, etc. Keep it simple for now

        dims_specs_to_use = self.dims_specs_options.get(op_str, self.dims_specs_options["default"])

        param_combinations_exhaustive = itertools.product(
            is_dynamic_options,          # input_dynamic_shape 2
            is_dynamic_options,          # output_is_dynamic_shape 2
            dims_specs_to_use,           # 6
            vector_rank_options,         # input_vector_rank 3
            self.simple_continuities     # input_continuity
        )
        return param_combinations_exhaustive

    def generate_test_name(self, datatype, input_is_dynamic, output_is_dynamic,
                           dims_spec, vector_rank, input_continuity):
        parts = []
        
        # 1. datatype
        parts.append(f"{datatype.name_suffix}")
        
        # 2. input dynamic/fixed
        input_shape_type = "dynamic" if input_is_dynamic else "fixed"
        parts.append(f"input_{input_shape_type}")
        
        # 3. output dynamic/fixed
        output_shape_type = "dynamic" if output_is_dynamic else "fixed"
        parts.append(f"output_{output_shape_type}")
        
        # 4. vector rank
        if vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{vector_rank}D_vector")
        
        # 5. continuity - contiguous->raw_tensor, non_contiguous->view
        if input_continuity.is_contiguous:
            parts.append("raw_tensor")
        else:
            op_str = "mul2" if input_continuity.big_tensor_op == "*2" else "add3" if input_continuity.big_tensor_op == "+3" else "add7"
            parts.append(f"view_dim{input_continuity.non_contiguous_dim}_{op_str}")
        
        # 6. shape information
        if dims_spec == [2, 3, 16, 16]:
            parts.append("shape1")
        elif dims_spec == [2, 1, 16, 7]:
            parts.append("shape2")
        else:
            # Fallback for other shapes
            shape_str = "x".join(map(str, dims_spec))
            parts.append(f"shape_{shape_str}")
        
        return "_".join(parts)

    

    def get_op_call_lines(self, ntt_op_str):
        """Generate NTT unary operation code"""
        return [
            "// Execute unary operation",
            f"ntt::unary<ntt::ops::{ntt_op_str}>(ntt_input, ntt_output);",
            ""
        ]

    def generate_ntt_output_to_test(self, datatype, op_str, 
                                input_is_dynamic, output_is_dynamic,
                                dims_spec, 
                                vec_rank, vec_param,
                                input_continuity):
        code = []

        code.append(f"{self.indent}//---init ntt_input---")
        tensor_init_lhs_code = self.generate_tensor_init(
            datatype = datatype, shape_type = input_is_dynamic, dim_spec = dims_spec,
            continuity = input_continuity, var_name="ntt_input",
            name_suffix = "", vector_rank = vec_rank,
            P = vec_param, integer_only = False
        )
        code.extend([f"{self.indent}{line}" for line in tensor_init_lhs_code])

        output_shape_expr = self.generate_shape_init(output_is_dynamic, dims_spec)
        op_call_lines = self.get_op_call_lines(op_str)
        output_element_type = self.get_element_cpp_type(datatype.cpp_type, vec_rank, vec_param)
        ntt_output_and_op_code = self.generate_ntt_output_and_op_section(
            datatype = datatype,
            output_shape_expr = output_shape_expr,
            cast_mode = 0,
            ntt_op_call_lines = op_call_lines,
            output_var_name = "ntt_output",
            output_element_type = output_element_type
        )
        code.extend([f"{self.indent}{line}" for line in ntt_output_and_op_code])


        return code, output_shape_expr, output_element_type

    def _generate_vector_to_scalar_conversion(
        self, code, 
        datatype, input_var_name, input_is_dynamic,
                input_dims_spec, vec_rank
    ):
        assert(vec_rank > 0)
    
        unsqueeze_dims = ""
        unpack_dims = ""
        scalar_dims = None
        # 1. unsqueeze
        if vec_rank == 1:
            unsqueeze_dims = f"{len(input_dims_spec)}"
            scalar_dims =  [str(d) for d in input_dims_spec] + ["P"]
            unpack_dims = unsqueeze_dims
        else: #vec_rank == 2
            unsqueeze_dims = f"{len(input_dims_spec)}, {len(input_dims_spec)+1}"
            scalar_dims = [str(d) for d in input_dims_spec] + ["4" ,"P"]
            unpack_dims = unsqueeze_dims

        code.append(f"auto {input_var_name}_unsqueezed = ({input_var_name}).unsqueeze(fixed_shape_v<{unsqueeze_dims}>);")

        # 2. unpack
        code.append(f"auto {input_var_name}_scalar = ntt::make_tensor<{datatype.cpp_type}>(fixed_shape_v<{','.join(map(str,scalar_dims))}>);")
        code.append(f"ntt::unpack({input_var_name}_unsqueezed, {input_var_name}_scalar, fixed_shape_v<{unpack_dims}>);")
        return scalar_dims

    def _prepare_double_scalar_input(
        self, code, datatype, input_is_dynamic, input_dims_spec, 
        vector_rank, vec_param, input_continuity, cast_target = "double"
    ):
        """
        1. ntt_tensor -> ntt_tensor_contiguous
        2. ntt_tensor_contiguous -> ntt_tensor_aligned_of_scalar
        2.1  ntt_tensor_contiguous of vector -> ntt_tensor_contiguous of scalar
        3.  ntt_tensor_aligned_of_scalar -> ntt_tensor_aligned_double
        """
        """
        input_var_name, input_copy_code = self._prepare_contiguous_input(
            "ntt_input", datatype, vector_rank, vec_param,
            input_is_dynamic, input_dims_spec, input_continuity
        )
        code.extend(input_copy_code)
        """
        input_var_name = "ntt_input"

        code.append(f"// align in NTT, then cast to {cast_target}, then process in ORT")

        input_is_vec = vector_rank > 0

        input_scalar_dims = input_dims_spec
        if input_is_vec:
            input_scalar_dims = self._generate_vector_to_scalar_conversion(
                code, datatype,  input_var_name, input_is_dynamic,
                input_dims_spec, vector_rank
            )
        else:
            code.append("// 1.1.b for input that are tensor of scalar")
            code.append(f"auto ntt_input_scalar = ({input_var_name}).view();")

        input_double_shape_expr = self.generate_shape_init(input_is_dynamic, input_scalar_dims)

        code.append(f"//1.2 get ntt_input_{cast_target}")
        code.append(f"auto ntt_input_{cast_target} = ntt::make_tensor<{cast_target}>({input_double_shape_expr});")
        code.append("")
        code.append(f"ntt::cast(ntt_input_scalar, ntt_input_{cast_target});")

        return input_scalar_dims


    def _execute_ort_operation(self, code, datatype, ntt_op_str, cast_target):
        """1. cast to ort tensor"""
        """2. calculate ort_output"""
        code.append("")
        code.append("// 2. calculated ort_output")
        code.append(f"auto ort_input = NttTest::ntt2ort(ntt_input_{cast_target});")
        code.extend(self.generate_ort_output(datatype, ntt_op_str))
        code.append(f"auto ort_golden_{cast_target} = ort_output;")
        return f"ort_golden_{cast_target}"



    def _generate_ntt_cast_golden_output(self, datatype, op_str, 
                                   input_is_dynamic, output_is_dynamic,
                                   dims_spec , vector_rank, vec_param, 
                                   input_continuity, cast_target):
        """Special handling for types that cannot be cast in ORT"""
        code = []

        self._prepare_double_scalar_input(
            code, datatype, input_is_dynamic, dims_spec,
            vector_rank, vec_param, input_continuity, cast_target)

        ort_golden_double_var = self._execute_ort_operation(
            code, datatype, op_str, cast_target
        )

        self._cast_ort_golden_double_into_ntt_shape(
            code, datatype, op_str, output_is_dynamic, dims_spec, vector_rank,
            vec_param, ort_golden_double_var, cast_target
        )
            
        return code

    def _generate_ort_cast_golden_output(self, datatype, op_str,
                                   input_is_dynamic, output_is_dynamic,
                                   dims_spec, vector_rank, vec_param,
                                   input_continuity, cast_target):
        """Generate golden output using ORT with optional casting"""
        code = []
        
        need_cast_in_ort = self._need_cast_in_ort(datatype, op_str)
        """
        input_var_name, input_copy_code = self._prepare_contiguous_input(
            "ntt_input", datatype, vector_rank, vec_param,
            input_is_dynamic, dims_spec, input_continuity
        )
        code.extend(input_copy_code)
        """
        input_var_name = "ntt_input"
        ntt2ort_input = input_var_name
        ort_input_origin_type = "ort_input_org"
        code.append(f"auto {ort_input_origin_type} = NttTest::ntt2ort({ntt2ort_input});")

        # this is not a good impmentation
        if need_cast_in_ort:
            code.append(f"auto ort_input = ortki_Cast({ort_input_origin_type}, 1, ortki::{self.ort_datatype_map[cast_target]});")
        else:
            code.append(f"auto ort_input = {ort_input_origin_type};")

        code.extend(self.generate_ort_output(datatype, op_str))

        if need_cast_in_ort:
            cast_to_orig_type_code = self.generate_ort_cast_back(datatype)
            code.extend(cast_to_orig_type_code)
        else:
            code.append(f"auto ort_golden = ort_output;")

        # Generate output shape and element type for ort2ntt conversion
        output_shape_expr = self.generate_shape_init(output_is_dynamic, dims_spec)
        output_element_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, vec_param)
        
        cast_code, golden_var_name = self.generate_ort_back2ntt(
            datatype,
            output_element_type,
            output_shape_expr,
            cast_mode=0,  # no cast be dealed in this step
            ntt_output_var_name="ntt_output",
            ort_output_var_name="ort_golden",
            ort_type = cast_target)
        code.extend(cast_code)
        return code, golden_var_name

    def _get_cast_target(self, datatype, op_str):
        if op_str == "cos":
            return "float"
        else:
            return "double"

    def generate_ntt_golden_output(self, datatype, op_str,
                                   input_is_dynamic, output_is_dynamic,
                                   dims_spec, vector_rank, vec_param,
                                   input_continuity):
        code = []
        
        # Check if datatype needs special fp8 handling
        need_cast_in_ntt = datatype.cpp_type in self.types_need_cast_in_ntt
        golden_var_name = "ntt_golden"
        cast_target = self._get_cast_target(datatype, op_str)

        if need_cast_in_ntt:
            # cast in ntt
            code.extend(self._generate_ntt_cast_golden_output(
                datatype, op_str, input_is_dynamic, output_is_dynamic,
                dims_spec, vector_rank, vec_param, input_continuity, cast_target
            ))
        else:
            # cast in ort or need not cast
            golden_output_code, golden_var_name = self._generate_ort_cast_golden_output(
                datatype, op_str, input_is_dynamic, output_is_dynamic,
                dims_spec, vector_rank, vec_param, input_continuity, cast_target
            )
            code.extend(golden_output_code)

        return code, golden_var_name


    def generate_test_case(
            self,
            datatype,
            op_str,
            input_is_dynamic,
            output_is_dynamic,
            dims_spec,
            vector_rank,
            input_continuity):
        test_name = self.generate_test_name(datatype, input_is_dynamic, output_is_dynamic,
                                             dims_spec, vector_rank, input_continuity)

        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"

        code: List[str] = []

        vec_param = "P" if vector_rank > 0 else None

        # 1. Test header and constants
        code.extend(self.generate_function_name(f"UnaryTest{op_str}", datatype, test_name))
        if vector_rank > 0:
            code.extend(self.generate_P_constants(P))


        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(
                                        datatype, op_str, 
                                        input_is_dynamic, output_is_dynamic,
                                        dims_spec, 
                                        vector_rank, vec_param,
                                        input_continuity)
        code.extend(ntt_output_code)

        golden_output_code, golden_var_name = self.generate_ntt_golden_output(
            datatype, op_str, input_is_dynamic, output_is_dynamic,
            dims_spec, vector_rank, vec_param, input_continuity
        )

        code.extend([f"{self.indent}{line}" for line in golden_output_code])

        # Add comparison code
        output_shape_expr = self.generate_shape_init(output_is_dynamic, dims_spec)
        output_element_type = self.get_element_cpp_type(datatype.cpp_type, vector_rank, vec_param)
        
        compare_code = self.generate_compare(
            ntt_output_var_name="ntt_output",
            golden_var_name=golden_var_name,
            ulp_tolerances=self._get_ulp_tolerance(op_str, datatype)
        )
        code.extend([f"{self.indent}{line}" for line in compare_code])

        return "\n".join(code)

    def generate_all_tests_for_type(self, datatype, op_str):
        # Generate tests for a specific data type and operation string
        code = []

        code.append(self.generate_header())

        if op_str in self.ort_custom_function:
            custom_op_func = self.generate_ort_custom_op(datatype, op_str)
            code.append(custom_op_func)

        param_combinations = self._get_param_combinations(op_str)

        for input_is_dynamic, output_is_dynamic, dims_spec, input_vector_rank, input_continuity in param_combinations:
            # Generate test cases for each combination of parameters
            test_case = self.generate_test_case(datatype, op_str, input_is_dynamic, output_is_dynamic, dims_spec, input_vector_rank, input_continuity)
            code.append(test_case)

        code.append(self.generate_footer())
        return "\n".join(code)

def is_uint(datatype):
    return "uint" in datatype.cpp_type

def is_int(datatype):
    return "int" in datatype.cpp_type

def need_skip(datatype, op_str):
    if(datatype.cpp_type == "bool"):
        return True
    if(is_uint(datatype) and op_str == "abs"):
        return True
    if(is_int(datatype) and (op_str == "cos" or op_str == "sin")):
        return True

    return False

def generate_tests_for_op(op_str, generator):
    for datatype in ALL_DATATYPES:

        if(need_skip(datatype, op_str)):
            continue


        if op_str in generator.OPRATOR_SPECICAL_TYPES:
            # For operations with special type requirements, use the specified data types
            special_types = generator.OPRATOR_SPECICAL_TYPES[op_str]
            # Check if current datatype's cpp_type is in the special types
            if datatype.cpp_type in special_types:
                datatype = special_types[datatype.cpp_type]

        test_code = generator.generate_all_tests_for_type(datatype, op_str)
        filename = f"test_ntt_unary_{datatype.name_suffix.lower()}_{op_str}_generated.cpp"
        output_filepath = os.path.join(generated_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)
    

if __name__ == "__main__":
    generator = UnaryTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (ctest) and then the generated subdirectory
    ctest_directory = os.path.dirname(script_directory)
    generated_directory = os.path.join(ctest_directory, "generated")
    
    # Ensure generated directory exists
    os.makedirs(generated_directory, exist_ok=True)
    
    generated_filenames = []  # collect all generated file names

    # for datatype in ALL_DATATYPES:
    #     test_code = generator.generate_all_tests_for_type(datatype)
    #     filename = f"test_ntt_unary_{datatype.name_suffix.lower()}_generated.cpp"
    #     output_filepath = os.path.join(generated_directory, filename)

    #     with open(output_filepath, "w") as f:
    #         f.write(test_code)
        
    #     print(f"Test file generated: {output_filepath}")
    #     generated_filenames.append(filename)

    for op_str in (generator.op_str_map_exhaustive.keys() | generator.op_str_map_simplified.keys()):
        generate_tests_for_op(op_str, generator)
    # Generate cmake list file in the generated directory
    generate_cmake_list(generated_directory, generated_filenames, "generated_unary_tests.cmake", "GENERATED_UNARY_TEST_SOURCES")
