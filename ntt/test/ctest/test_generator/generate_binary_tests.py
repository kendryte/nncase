import itertools
import os
from typing import List
from test_generator_base import *



class BinaryTestGenerator(BaseTestGenerator):
    def __init__(self):
        super().__init__()
        
        # ORT *binary operations* do not support these data types, need to cast to double 
        # fortunately, they could be *cast* in ort( fp8 are unfortunate)
        self.types_need_cast_in_ort = {
            "swishb": [ 'bool', 'uint8_t', 'uint16_t', 'uint32_t',
                    'uint64_t', 'int8_t', 'int16_t', 'bfloat16', 'half',
                      "int32_t", "int64_t",
                    'float_e4m3_t', 'float_e5m2_t' 
                   ],

            "default": [ 'bool',  'int8_t', 'int16_t', 'bfloat16', 'half',
                    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
                ]
        }
        self.types_need_cast_in_ntt = {
            'float_e4m3_t', 'float_e5m2_t' 
        }

        self.dims_specs_options = {
            "swishb":   [
                # Scalar broadcast
                ([2, 3, 16, 16], [1])
            ],
            
            "default": [
                # No broadcast
                ([2, 3, 16, 16], [2, 3, 16, 16]),
                # Scalar broadcast
                ([1], [2, 3, 16, 16]),
                ([2, 3, 16, 16], [1]),
                # Vector broadcast
                ([16], [2, 3, 16, 16]),
                ([2, 3, 16, 16], [16]),
                # Multidirectional broadcast
                ([2, 1, 16, 1], [1, 3, 1, 16]),
            ]
        }
        
        
        # Define power operand ranges
        self.ALL_POW_OPRANDS = {
            "uint8_t": {"lhs_min": "0", "lhs_max": "3", "rhs_min": "0", "rhs_max": "3"},
            "int8_t": {"lhs_min": "-2", "lhs_max": "2", "rhs_min": "-3", "rhs_max": "3"},
            "int16_t": {"lhs_min": "-7", "lhs_max": "8", "rhs_min": "-4", "rhs_max": "4"},
            "uint16_t": {"lhs_min": "0", "lhs_max": "8", "rhs_min": "0", "rhs_max": "4"},
            "int32_t": {"lhs_min": "-15", "lhs_max": "15", "rhs_min": "-7", "rhs_max": "7"},
            "uint32_t": {"lhs_min": "0", "lhs_max": "15", "rhs_min": "0", "rhs_max": "7"},
            "int64_t": {"lhs_min": "0", "lhs_max": "15", "rhs_min": "-14", "rhs_max": "14"},
            "uint64_t": {"lhs_min": "0", "lhs_max": "15", "rhs_min": "0", "rhs_max": "14"},


            "float_e4m3_t": {"lhs_min": "float_e4m3_t(-3.0)", "lhs_max": "float_e4m3_t(2.0)", "rhs_min": "float_e4m3_t(-2.0f)", "rhs_max": "float_e4m3_t(3.0f)"},
            "float_e5m2_t": {"lhs_min": "float_e5m2_t(-3.0)", "lhs_max": "float_e5m2_t(3.0)", "rhs_min": "float_e5m2_t(-3.0f)", "rhs_max": "float_e5m2_t(3.0f)"},
            "bfloat16": {"lhs_min": "bfloat16(-64.0)", "lhs_max": "bfloat16(64.0)", "rhs_min": "-10.0_bf16", "rhs_max": "10.0_bf16"},
            "half": {"lhs_min": "half(-32.0)", "lhs_max": "half(32.0)", "rhs_min": "half(-3.0)", "rhs_max": "half(3.0)"},
            "float": {"lhs_min": "-256.0", "lhs_max": "256.0", "rhs_min": "-15.0", "rhs_max": "15.0"},
            "double": {"lhs_min": "-1000.0", "lhs_max": "1000.0", "rhs_min": "-50.0", "rhs_max": "50.0"},

        }



        self.integer_types = ['int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'] 
        
        self.ort_custom_function = {
            "ceil_div": self._generate_ort_ceil_div_function,
            "swishb": self._generate_ort_SwishB,
            "inner_product": self._generate_inner_product_operation,
        }

        # tests in self.op_str_map_exhaustive would go through exhaustive tests.
        self.op_str_map_exhaustive = {
            "add": f"auto ort_output = ortki_Add(ort_input_lhs, ort_input_rhs);",
            "swishb":  f"auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);",
            "inner_product":  \
                            "static bool element_is_vec = ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;\n" \
                            "   auto ort_output = ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec); " ,
            "outer_product":  \
                            "   auto ort_output =ortki_Mul(ort_input_lhs, ort_input_rhs); " 
        }
        self.op_str_map_simplified = {
            "sub": f"auto ort_output = ortki_Sub(ort_input_lhs, ort_input_rhs);",
            "mul": f"auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);",
            "div": f"auto ort_output = ortki_Div(ort_input_lhs, ort_input_rhs);",
            "ceil_div": "auto ort_output = ort_ceil_div(ort_input_lhs, ort_input_rhs);",
            "mod": f"auto ort_output = ortki_Mod(ort_input_lhs, ort_input_rhs, 1);",
            "min":  self._generate_minmax_operation("ortki_Min"),
            "max":  self._generate_minmax_operation("ortki_Max"),
            "pow": f"auto ort_output = ortki_Pow(ort_input_lhs, ort_input_rhs);",
            "floor_mod": lambda datatype: \
                "auto ort_output = ortki_Mod(ort_input_lhs, ort_input_rhs, 0);" \
                if datatype.cpp_type in self.integer_types and datatype.cpp_type not in self.types_need_cast_in_ort["default"] \
                else "auto ort_output = ortki_Sub(ort_input_lhs, ortki_Mul(ortki_Floor(ortki_Div(ort_input_lhs, ort_input_rhs)), ort_input_rhs));",
        }

        self.ulp_tolerances  = {
            "pow": {
                "default": 4
            },
            "default": {
                "default": 1
            }
        }



    def _generate_minmax_operation(self, operation_func):
        """Generate code for min/max operations with reduced duplication"""
        return (
            "const size_t num_inputs = 2;\n"
            "    ortki::OrtKITensor* input_tensors[num_inputs];\n"
            "    input_tensors[0] = ort_input_lhs;\n"
            "    input_tensors[1] = ort_input_rhs;\n"
            f"    auto ort_output = {operation_func}(input_tensors, num_inputs);"
        )

    def _generate_ceil_div_operation(self, datatype):
        """Generate code for ceil_div operation with reduced duplication"""
        # Determine the appropriate type and value for neg1
        types_to_cast = self.types_need_cast_in_ort.get(op_str, self.types_need_cast_in_ort["default"])
        if datatype.cpp_type == "int64_t":
            var_type = "int64_t"
            value_str = "-1"
        elif datatype.cpp_type not in types_to_cast: 
            # Now only int32_t in this case
            var_type = "int32_t"
            value_str = "-1"
        else:
            var_type = "double"
            value_str = "-1.0f"
        
        # Return the common template with variable substitution
        return (
            f"auto ntt_neg1 = make_tensor<{var_type}>(ntt::fixed_shape_v<1>);\n"
            f"    ntt_neg1(0) = {value_str};\n"
            "    auto ort_neg1 = NttTest::ntt2ort(ntt_neg1);\n"
            "    auto ort_output = ortki_Div(ortki_Add(ort_input_lhs, ortki_Add(ort_input_rhs, ort_neg1)), ort_input_rhs);"
        )

    def _generate_ort_const_var_info(self, datatype, const_value, op_str):
        """Generate variable type and value string for ORT constants"""
        # !!! Very ugly, must be refactored later
        types_to_cast = self.types_need_cast_in_ort.get(op_str, self.types_need_cast_in_ort["default"])
        if not "int" in datatype.cpp_type: # float
            if datatype.cpp_type in types_to_cast:
                var_type = "double"
                value_str = f"{const_value}.0f"
            else:
                var_type = datatype.cpp_type
                value_str = f"static_cast<{datatype.cpp_type}>({const_value})"
        else: # uintx, intx
            if(op_str == "pow"):
            # Ortki can not take int as exp input
                var_type = "double"
                value_str = f"{const_value}.0f"
            else:
                if( datatype.cpp_type in types_to_cast):
                    var_type = "double"
                    value_str = f"{const_value}.0f"
                else:
                    var_type = datatype.cpp_type
                    value_str = f"static_cast<{datatype.cpp_type}>({const_value})"
        return var_type, value_str

    def _generate_ort_ceil_div_function(self, datatype):
        """Generate the ort_ceil_div function definition"""
        const_var_type, const_value_str = self._generate_ort_const_var_info(datatype, -1, "ceil_div")
        
        return (
            f"static ortki::OrtKITensor* ort_ceil_div(ortki::OrtKITensor* ort_input_lhs, ortki::OrtKITensor* ort_input_rhs) {{\n"
            f"    auto ntt_neg1 = make_tensor<{const_var_type}>(ntt::fixed_shape_v<1>);\n"
            f"    ntt_neg1(0) = {const_value_str};\n"
            "    auto ort_neg1 = NttTest::ntt2ort(ntt_neg1);\n"
            "    return ortki_Div(ortki_Add(ort_input_lhs, ortki_Add(ort_input_rhs, ort_neg1)), ort_input_rhs);\n"
            "}\n\n"
        )
    def _generate_inner_product_operation(self, datatype):
        """Generate the ortki_inner_product function definition"""
        return (
        "static ortki::OrtKITensor* ortki_inner_product(ortki::OrtKITensor* ort_input_lhs, ortki::OrtKITensor* ort_input_rhs, bool  element_is_vec) {\n"
        "   ortki::OrtKITensor* product_tensor = ortki_Mul(ort_input_lhs, ort_input_rhs);\n"
        "   if (!element_is_vec)\n"
        "       return product_tensor;\n"
        "   int64_t axis_data[] = {-1};                         \n"
        "   const int64_t axis_shape[] = {1};                   \n"
        "   size_t axis_rank = 1;                               \n"
        "   auto ort_type = nncase::NttTest::primitive_type2ort_type<int64_t>();\n"
        "   ortki::OrtKITensor* axes_tensor = make_tensor(\n"
        "       axis_data,                                       // void* buffer\n"
        "       ort_type,\n"
        "       axis_shape,                                      // const int64_t* shape\n"
        "       axis_rank                                        // rank\n"
        "   );\n"
        "   if (axes_tensor == nullptr) {\n"
        "       return nullptr;\n"
        "   }\n"
        "   int64_t keepdims = 0;\n"
        "   int64_t noop_with_empty_axes = 0;\n"
        "   ortki::OrtKITensor* result_tensor = ortki_ReduceSum(\n"
        "       product_tensor,\n"
        "       axes_tensor,\n"
        "       keepdims,\n"
        "       noop_with_empty_axes);\n"
        "   return result_tensor;\n"
        "}"
        )

    def _generate_ort_SwishB(self, datatype):
        """Generate the ortki_SwishB function definition"""
        const_var_type, const_value_str = self._generate_ort_const_var_info(datatype, 1, "swishb")
        
        return (
            f"static ortki::OrtKITensor* ortki_SwishB(ortki::OrtKITensor* ort_input, ortki::OrtKITensor* beta_tensor) {{\n"
            f"    auto ntt_1_tensor = make_tensor<{const_var_type}>(ntt::fixed_shape_v<1>);\n"
            f"    ntt_1_tensor(0) = {const_value_str};\n"
            "    auto ort_1 = NttTest::ntt2ort(ntt_1_tensor);\n"           
            "    auto ort_neg = ortki_Neg(ort_input);\n"
            "    auto ort_mul = ortki_Mul(ort_neg, beta_tensor);\n"
            "    auto ort_exp = ortki_Exp(ort_mul);\n"
            "    auto ort_add = ortki_Add(ort_1, ort_exp);\n"
            "    return ortki_Div(ort_input, ort_add);\n"
            "}\n\n"
        )


    def _generate_aligned_ntt_scalar_input(self, ntt_op_str, datatype, input_var_name, var_name, 
                                   is_dynamic_shape, dims_spec, vector_rank, other_vector_rank):
        """Generate aligned NTT input tensors for fp8 operations"""
        """ normal case: tensor<vector<P>, axbxc> -> tensor<scalar, axbxcxP>
            aligned case: tensor<vector<P>, axbxc> align with tensor of 2D vector->  tensor<scalar, axbxcx1xP>
        """
        code = []
        aligned_dims = None
        
        # Determine if tensors need alignment based on vector ranks
        need_alignment = vector_rank + other_vector_rank > 0

        #fistly unsqueeze
        
        unsqueeze_dims = ""
        unpack_dims = ""
        if vector_rank >= other_vector_rank :

            if vector_rank == 0:
                code.append(f"// for tensors pair that are all tensor of scalar")
                code.append(f"auto {input_var_name}_unsqueezed = {var_name};")
                aligned_dims = dims_spec
            else:
                if vector_rank == 1:
                    if ntt_op_str == "outer_product":
                        unsqueeze_dims = f"{len(dims_spec)}, {len(dims_spec)+1}"
                        if "lhs" in input_var_name:
                            unpack_dims = f"{len(dims_spec)}"
                            aligned_dims = [str(d) for d in dims_spec] + ["P", "1"]
                        elif "rhs" in input_var_name:
                            unpack_dims = f"{len(dims_spec)+1}"
                            aligned_dims = [str(d) for d in dims_spec] +["1", "P"]
                    else:
                        unsqueeze_dims = f"{len(dims_spec)}"
                        aligned_dims = [str(d) for d in dims_spec] + ["P"]
                        unpack_dims = unsqueeze_dims
                elif vector_rank == 2:
                    unsqueeze_dims = f"{len(dims_spec)}, {len(dims_spec)+1}"
                    aligned_dims = [str(d) for d in dims_spec] + ["4" ,"P"]
                    unpack_dims = unsqueeze_dims
                code.append(f"auto {input_var_name}_unsqueezed = ({var_name}).unsqueeze(fixed_shape_v<{unsqueeze_dims}>);")
        else:
            # vector_rank would be 0 or 1
            diff_rank = other_vector_rank - vector_rank
            if other_vector_rank == 1:
                # this vector rank must be 0
                unsqueeze_dims = f"{len(dims_spec)}"
                unpack_dims = unsqueeze_dims    
            else: # other_vector_rank == 2
                # this vector rank should be 1 or 0
                unsqueeze_dims = f"{len(dims_spec)}, {len(dims_spec)+1}"
                unpack_dims = f"{len(dims_spec) + 1}"
            code.append(f"auto {input_var_name}_unsqueezed = ({var_name}).unsqueeze(fixed_shape_v<{unsqueeze_dims}>);")
            aligned_dims = [str(d) for d in dims_spec] + ["1"] * (other_vector_rank-1)
            aligned_dims.append("P" if vector_rank == 1 else "1")

        if(vector_rank == 0 ):
            code.append(f"auto {input_var_name}_aligned = ({input_var_name}_unsqueezed).view();")
        else:
            code.append(f"auto {input_var_name}_aligned = ntt::make_tensor<{datatype.cpp_type}>(fixed_shape_v<{','.join(map(str, aligned_dims))}>);")
            code.append(f"ntt::unpack({input_var_name}_unsqueezed, {input_var_name}_aligned, fixed_shape_v<{unpack_dims}>);")
        return code, aligned_dims

    def _prepare_double_input_alignment(self, code, datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                   lhs_dims_spec, rhs_dims_spec, lhs_vector_rank, rhs_vector_rank,
                                   lhs_continuity, rhs_continuity, lhs_vec_param, rhs_vec_param,
                                   ntt_op_str):
        """Prepare input alignment and cast to double for ORT operations
        1. ntt_tensor -> ntt_tensor_contiguous
        2. ntt_tensor_contiguous -> ntt_tensor_aligned_of_scalar
        2.1  ntt_tensor_contiguous of vector -> ntt_tensor_contiguous of scalar
        3.  ntt_tensor_aligned_of_scalar -> ntt_tensor_aligned_double
        """
        # 1.1 get ntt_input_lhs_aligned_{cpp.type}_scalar, ntt_input_rhs_aligned
        # Prepare contiguous inputs
        """
        lhs_var_name, lhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_lhs", datatype, lhs_vector_rank, lhs_vec_param,
            lhs_is_dynamic_shape, lhs_dims_spec, lhs_continuity
        )
        code.extend(lhs_copy_code)
        
        rhs_var_name, rhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_rhs", datatype, rhs_vector_rank, rhs_vec_param,
            rhs_is_dynamic_shape, rhs_dims_spec, rhs_continuity
        )
        code.extend(rhs_copy_code)
        
        """
        lhs_var_name = "ntt_input_lhs"
        rhs_var_name = "ntt_input_rhs"
        code.append("// align in NTT, then cast to double, then process in ORT")
        
        # Determine if tensors need alignment based on vector ranks
        need_alignment = (lhs_vector_rank + rhs_vector_rank != 0 )
        # Initialize aligned dimensions to default values
        lhs_aligned_dims = lhs_dims_spec
        rhs_aligned_dims = rhs_dims_spec
        
        if need_alignment:
            # 1.1.a for tensors pair that one of which is tensor of vector
            # Generate aligned lhs input
            lhs_code, lhs_aligned_dims = self._generate_aligned_ntt_scalar_input(
                ntt_op_str, datatype, "ntt_input_lhs", lhs_var_name, lhs_is_dynamic_shape, 
                lhs_dims_spec, lhs_vector_rank, rhs_vector_rank)
            code.extend(lhs_code)
                
            # Generate aligned rhs input
            rhs_code, rhs_aligned_dims = self._generate_aligned_ntt_scalar_input(
                ntt_op_str, datatype, "ntt_input_rhs", rhs_var_name, rhs_is_dynamic_shape,
                rhs_dims_spec, rhs_vector_rank, lhs_vector_rank)
            code.extend(rhs_code)
        else:
        # 1.1.b for tensors pair that are all tensor of scalar
            code.append("// 1.1.b for tensors pair that are all tensor of scalar")
            code.append(f"auto ntt_input_lhs_aligned = ({lhs_var_name}).view();")
            code.append(f"auto ntt_input_rhs_aligned = ({rhs_var_name}).view();")
        
        # 1.2 get ntt_lhs/rhs_double
        lhs_double_shape_expr = self.generate_shape_init(lhs_is_dynamic_shape, lhs_aligned_dims)
        rhs_double_shape_expr = self.generate_shape_init(rhs_is_dynamic_shape, rhs_aligned_dims)
        
        code.append(f"// 1.2 get ntt_lhs/rhs_double")
        code.append(f"auto ntt_lhs_double = ntt::make_tensor<double>({lhs_double_shape_expr});")
        code.append(f"auto ntt_rhs_double = ntt::make_tensor<double>({rhs_double_shape_expr});")
        code.append("")
        code.append("ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);")
        code.append("ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);")
        
        return lhs_aligned_dims, rhs_aligned_dims

    def _execute_ort_operation(self, code, datatype, ntt_op_str):
        """Execute ORT operation on aligned double tensors"""
        """1. cast to ort tensor"""
        """2. calculate ort_output"""
        # 2. calculated ort_output
        code.append("")
        code.append("// 2. calculated ort_output")
        code.append(f"auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_lhs_double, ntt_rhs_double, false, false);")
        code.extend(self.generate_ort_output(datatype, ntt_op_str))
        code.append(f"auto ort_golden_double = ort_output;")
        return "ort_golden_double"



    def _generate_ntt_cast_golden_output(self, datatype, 
                                   lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                   lhs_dims_spec, rhs_dims_spec,
                                   lhs_vector_rank, rhs_vector_rank,
                                   lhs_continuity, rhs_continuity,
                                   lhs_vec_param, rhs_vec_param,
                                   ntt_op_str):
        """Special handling for types that cannot be cast in ORT"""
        """following steps would be done if necessary:
        1. ntt_tensor -> ntt_tensor_contiguous
        2. ntt_tensor_contiguous -> ntt_tensor_aligned_of_scalar
        2.1  ntt_tensor_contiguous of vector -> ntt_tensor_contiguous of scalar
        2.2  add proper dim_1 to the shape of ntt_tensor_contiguous of scalar
        3.  ntt_tensor_aligned_of_scalar -> ntt_tensor_aligned_double
        4. transform to ort tensor and performe the opertation.
        5. transform back to ntt tensor of double scalar
        6. cast back to original type, still tensor of scalar
        7. vectorized back to original tensor of vector (if necessary)
        """

        code = []
        
        # 1. Prepare input alignment and cast to double
        lhs_aligned_dims, rhs_aligned_dims = self._prepare_double_input_alignment(
            code, datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec, lhs_vector_rank, rhs_vector_rank,
            lhs_continuity, rhs_continuity, lhs_vec_param, rhs_vec_param,
            ntt_op_str
        )
        
        # 2. Execute ORT operation
        ort_golden_double_var = self._execute_ort_operation(
            code, datatype, ntt_op_str
        )
        
        # Calculate output shape for scalar tensor
        output_is_dynamic_shape, output_dims_spec = self.get_binary_output_shape(
            lhs_is_dynamic_shape, rhs_is_dynamic_shape, lhs_dims_spec, rhs_dims_spec)
        output_vector_rank = self._get_output_vector_rank(ntt_op_str, lhs_vector_rank, rhs_vector_rank)
        output_vec_param = self._get_output_vec_param(ntt_op_str, lhs_vec_param, rhs_vec_param)


        # 3. Process ORT output back to NTT format
        self._cast_ort_golden_double_into_ntt_shape(
            code, datatype, ntt_op_str, output_is_dynamic_shape, 
            output_dims_spec, output_vector_rank, 
            output_vec_param, ort_golden_double_var
        )
        
        return code

    def _generate_ort_cast_golden_output(self, datatype, 
                                   lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                   lhs_dims_spec, rhs_dims_spec,
                                   lhs_vector_rank, rhs_vector_rank,
                                   lhs_continuity, rhs_continuity,
                                   lhs_vec_param, rhs_vec_param,
                                   output_element_type, output_shape_expr,
                                   ntt_op_str):
        """
        Generate golden output using ORT with optional casting
        1. get ntt_input_continuity
        2. get ort_input
        3. if need cast, cast ort_input to ort_input double
        4. perform operation
        """
        code = []
        
        # Check if datatype needs to be cast to float32
        need_cast_in_ort = self._need_cast_in_ort(datatype, ntt_op_str)

        lhs_var_name, lhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_lhs", datatype, lhs_vector_rank, lhs_vec_param,
            lhs_is_dynamic_shape, lhs_dims_spec, lhs_continuity
        )
        code.extend(lhs_copy_code)
        ntt2ort_lhs = lhs_var_name

        rhs_var_name, rhs_copy_code = self._prepare_contiguous_input(
            "ntt_input_rhs", datatype, rhs_vector_rank, rhs_vec_param,
            rhs_is_dynamic_shape, rhs_dims_spec, rhs_continuity
        )
        code.extend(rhs_copy_code)
        ntt2ort_rhs = rhs_var_name

        if need_cast_in_ort:
            code.append("// ort_input_lhs, ort_input_rhs would be tensor of double in ort format")

            code.append("")

        need_cast_str = "true" if need_cast_in_ort else "false"
        is_outer_product = "true" if ntt_op_str == "outer_product" else "false"

        code.extend([f"auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort({ntt2ort_lhs},{ntt2ort_rhs}, {need_cast_str}, {is_outer_product});"])
 
        code.extend(self.generate_ort_output(datatype, ntt_op_str))

        if need_cast_in_ort:
            cast_to_orig_type_code = self.generate_ort_cast_back(datatype)
            code.extend(cast_to_orig_type_code)
        else:
            code.append(f"auto ort_golden = ort_output;")
            
        
        cast_code, golden_var_name = self.generate_ort_back2ntt(
            datatype,
            output_element_type,
            output_shape_expr,
            cast_mode= 0, # no cast be dealed in this step
            ntt_output_var_name="ntt_output",
            ort_output_var_name="ort_golden")
        code.extend(cast_code)
        return code, golden_var_name

    def is_div_operation(self) -> bool:
        """Check if the current operation is division, to disable zero generation."""
        result = (hasattr(self, 'ntt_op_str') and self.ntt_op_str in ["div", "mod", "floor_mod", "ceil_div"])
        return result

    def generate_test_name(self, datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape, 
        lhs_dims_spec, rhs_dims_spec, 
        lhs_vector_rank, rhs_vector_rank, 
        lhs_continuity, rhs_continuity, test_name_suffix):
        
        parts = []
        
        #1. datatype
        parts.append(f"{datatype.name_suffix}")
        
        # 2.  lhs dynamic
        lhs_shape_type = "dynamic" if lhs_is_dynamic_shape else "fixed"
        parts.append(f"lhs_{lhs_shape_type}")
        
        # lhs vector rank
        if lhs_vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{lhs_vector_rank}D_vector")
        
        #  contiguous->view, non_contiguous->raw_tensor
        if lhs_continuity.is_contiguous:
            parts.append("raw_tensor")
        else:
            op_str = "mul2" if lhs_continuity.big_tensor_op == "*2" else "add3" if lhs_continuity.big_tensor_op == "+3" else "add7"
            parts.append(f"view_{lhs_continuity.non_contiguous_dim}_{op_str}")
        
        # 3. rhs
        rhs_shape_type = "dynamic" if rhs_is_dynamic_shape else "fixed"
        parts.append(f"rhs_{rhs_shape_type}")
        
        # rhs vector rank
        if rhs_vector_rank == 0:
            parts.append("scalar")
        else:
            parts.append(f"{rhs_vector_rank}D_vector")
        
        #  continuity
        if rhs_continuity.is_contiguous:
            parts.append("raw_tensor")
        else:
            op_str = "mul2" if rhs_continuity.big_tensor_op == "*2" else "add3" if rhs_continuity.big_tensor_op == "+3" else "add7"
            parts.append(f"view_dim{rhs_continuity.non_contiguous_dim}_{op_str}")
        
        # 4. braodcast type
        if lhs_dims_spec == rhs_dims_spec:
            broadcast_info = "no_broadcast"
        elif lhs_dims_spec == [1]:
            broadcast_info = "lhs_singleton_broadcast"  # [1] 表示单元素广播
        elif rhs_dims_spec == [1]:
            broadcast_info = "rhs_singleton_broadcast"  # [1] 表示单元素广播
        elif len(lhs_dims_spec) == 1 and len(rhs_dims_spec) > 1:
            broadcast_info = "lhs_1d_broadcast"  # 左操作数是一维张量广播
        elif len(rhs_dims_spec) == 1 and len(lhs_dims_spec) > 1:
            broadcast_info = "rhs_1d_broadcast"  # 右操作数是一维张量广播
        else:
            broadcast_info = "multi_broadcast"  # 多维广播
            
        parts.append(broadcast_info)
        if test_name_suffix:
            parts.append(test_name_suffix)
        
        return "_".join(parts)

    def get_binary_output_shape(self, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                lhs_shape, rhs_shape):
        output_is_dynamic_shape = lhs_is_dynamic_shape or rhs_is_dynamic_shape

        if len(lhs_shape) < len(rhs_shape):
            shorter_shape, longer_shape = lhs_shape, rhs_shape
        else:
            shorter_shape, longer_shape = rhs_shape, lhs_shape

        # Prepend 1s to the shorter shape to match the rank of the longer shape for broadcasting.
        rank_diff = len(longer_shape) - len(shorter_shape)
        padded_shorter_shape = [1] * rank_diff + shorter_shape
        
        # Check for broadcasting compatibility.
        for dim1, dim2 in zip(longer_shape, padded_shorter_shape):
            assert dim1 == dim2 or min(dim1, dim2) == 1, \
                f"Shapes {lhs_shape} and {rhs_shape} are not broadcast-compatible"
        
        # The output shape is the element-wise maximum of the two shapes.
        output_shape = [max(dim1, dim2) for dim1, dim2 in zip(longer_shape, padded_shorter_shape)]
        
        return output_is_dynamic_shape, output_shape


    def get_op_call_lines(self, ntt_op_str):
        """Generate NTT binary operation code"""
        return [
            "// Execute binary operation",
            f"ntt::binary<ntt::ops::{ntt_op_str}>(ntt_input_lhs, ntt_input_rhs, ntt_output);",
            ""
        ]



    def _get_output_vector_rank(self, ntt_op_str, lhs_vector_rank, rhs_vector_rank):
        """Determine the output vector rank based on the operation type and input ranks."""
        if ntt_op_str == "inner_product":
            return 0
        elif ntt_op_str == "outer_product":
            if lhs_vector_rank == 0 and rhs_vector_rank == 0:
                return 0
            elif lhs_vector_rank == 1 or rhs_vector_rank == 1:
                return 2
        else:
            return max(lhs_vector_rank, rhs_vector_rank)
        
    def _get_output_vec_param(self, ntt_op_str, lhs_vec_param, rhs_vec_param):
        """Determine the output pack parameter based on the operation type and input pack parameters."""
        if ntt_op_str == "outer_product":
            # For outer_product, return a tuple of both pack parameters
            return (lhs_vec_param, rhs_vec_param)
        else:
            return lhs_vec_param if lhs_vec_param else rhs_vec_param
    def generate_ntt_golden_output(self, datatype, 
                                    lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                    lhs_dims_spec, rhs_dims_spec,
                                    lhs_vector_rank, rhs_vector_rank,
                                    lhs_continuity, rhs_continuity,
                                    lhs_vec_param, rhs_vec_param,
                                    output_element_type, output_shape_expr,
                                    ntt_op_str):
        code = []
        
        # Check if datatype needs special fp8 handling
        need_cast_in_ntt = datatype.cpp_type in self.types_need_cast_in_ntt
        golden_var_name = "ntt_golden"
        
        if need_cast_in_ntt:
            # Special handling for fp8 types that cannot be cast in ORT
            code.extend(self._generate_ntt_cast_golden_output(
                datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                lhs_dims_spec, rhs_dims_spec, lhs_vector_rank, rhs_vector_rank,
                lhs_continuity, rhs_continuity, lhs_vec_param, rhs_vec_param, ntt_op_str
            ))
        else:
            # Original logic for non-fp8 types
            golden_output_code, golden_var_name = self._generate_ort_cast_golden_output(
                datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                lhs_dims_spec, rhs_dims_spec, lhs_vector_rank, rhs_vector_rank,
                lhs_continuity, rhs_continuity, lhs_vec_param, rhs_vec_param,
                output_element_type, output_shape_expr,
                ntt_op_str
            )
            code.extend(golden_output_code)

        return code, golden_var_name

    def generate_ntt_output_to_test(self, lhs_datatype, rhs_datatype,
                                    lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                                    lhs_dims_spec, rhs_dims_spec,
                                    lhs_vector_rank, rhs_vector_rank,
                                    lhs_continuity, rhs_continuity,
                                    lhs_vec_param, rhs_vec_param,
                                    ntt_op_str):
        indent = "    "
        code = []
        datatype = lhs_datatype  # Assume same datatype for both inputs
        # generate ntt_input_lhs, ntt_input_rhs, ntt_output
        code.append(f"{indent}//---init ntt_input_lhs---")
        tensor_init_lhs_code = self.generate_tensor_init( datatype=lhs_datatype,
            shape_type=lhs_is_dynamic_shape, dim_spec=lhs_dims_spec,
            continuity=lhs_continuity, var_name="ntt_input_lhs",
            name_suffix="_lhs", vector_rank=lhs_vector_rank,
            P=lhs_vec_param, integer_only= lhs_datatype.integer_only)
        code.extend([f"{indent}{line}" for line in tensor_init_lhs_code])

        code.append(f"{indent}//---init ntt_input_rhs---")
        tensor_init_rhs_code = self.generate_tensor_init( datatype=rhs_datatype,
            shape_type=rhs_is_dynamic_shape, dim_spec=rhs_dims_spec,
            continuity=rhs_continuity, var_name="ntt_input_rhs",
            name_suffix="_rhs", vector_rank=rhs_vector_rank,
            P=rhs_vec_param, integer_only= rhs_datatype.integer_only)
        code.extend([f"{indent}{line}" for line in tensor_init_rhs_code])

        output_is_dynamic_shape, output_dims_spec = self.get_binary_output_shape(
            lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec)
        
        output_vector_rank = self._get_output_vector_rank( ntt_op_str, lhs_vector_rank, rhs_vector_rank)
        code.append(f"{indent}//---generate output tensor---")

        output_shape_expr = self.generate_shape_init(output_is_dynamic_shape, output_dims_spec)
        # For binary ops, output vector rank matches inputs. Assume lhs.
        output_vec_param =  self._get_output_vec_param(ntt_op_str, lhs_vec_param, rhs_vec_param)
        output_element_type = self.get_element_cpp_type(datatype.cpp_type, output_vector_rank, output_vec_param)

        output_op_call_lines = self.get_op_call_lines(ntt_op_str)
        ntt_output_and_op_code = self.generate_ntt_output_and_op_section(
            datatype=datatype,
            output_shape_expr=output_shape_expr,
            cast_mode=0,  # Placeholder for now
            ntt_op_call_lines=output_op_call_lines,
            output_var_name="ntt_output",
            output_element_type=output_element_type
        )
        code.extend([f"{indent}{line}" for line in ntt_output_and_op_code])
        return code, output_shape_expr, output_element_type




    # lhs_dynamic: bool, lhs is dynamic or fixed
    # rhs_dynamic: bool, rhs is dynamic or fixed
    # lhs_shape: list[int], lhs shape, [1, 77, 3]
    # rhs_shape: list[int], rhs shape, [1, 77, 3]
    # braodcast_ways: list[int], broadcast ways, 0: no_broadcast 1: lhs_to_rhs, 2: rhs_to_lhs, 
    # lhs_vector_ranks: list[int], lhs vector ranks, 0, 1, 2
    # rhs_vector_ranks: list[int], rhs vector ranks, 0, 1, 2, 3
    # lhs_tensor: list[int], lhs is tensor or view, 0: tensor, 1: view
    # rhs_tensor: list[int], rhs is tensor or view
    def generate_test_case(
            self,
            lhs_datatype,
            rhs_datatype,
            lhs_is_dynamic_shape: bool,
            rhs_is_dynamic_shape: bool,
            lhs_dims_spec: List[int],
            rhs_dims_spec: List[int],
            lhs_vector_rank: int,
            rhs_vector_rank: int,
            lhs_continuity: Continuity,
            rhs_continuity: Continuity,
            ntt_op_str, test_name_suffix=None):
        # only support same datatype but different range now
        assert lhs_datatype.cpp_type == rhs_datatype.cpp_type
        
        datatype = lhs_datatype
        self.ntt_op_str = ntt_op_str  # Store operation type for allow_zr check

        test_name = self.generate_test_name(datatype, lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec,
            lhs_vector_rank, rhs_vector_rank,
            lhs_continuity, rhs_continuity, test_name_suffix)


        P = f"NTT_VLEN / (sizeof({datatype.cpp_type}) * 8)"
        code: List[str] = []
        lhs_vec_param = "P" if lhs_vector_rank > 0 else None
        rhs_vec_param = "P" if rhs_vector_rank > 0 else None

        # 1. Test header and constants
        code.extend(self.generate_function_name(f"BinaryTest{ntt_op_str}", datatype, test_name))
        if lhs_vector_rank > 0 or rhs_vector_rank > 0:
            code.extend(self.generate_P_constants(P))

        # # Generate output to test in ntt format
        ntt_output_code, output_shape_expr, output_element_type = self.generate_ntt_output_to_test(
                            lhs_datatype, rhs_datatype,
                            lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                            lhs_dims_spec, rhs_dims_spec,
                            lhs_vector_rank, rhs_vector_rank,
                            lhs_continuity, rhs_continuity,
                            lhs_vec_param, rhs_vec_param,
                            ntt_op_str)
        code.extend(ntt_output_code)


        # Generate golden output in ntt_tensor
        golden_output_code, golden_var_name = self.generate_ntt_golden_output(datatype,lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec,
            lhs_vector_rank, rhs_vector_rank,
            lhs_continuity, rhs_continuity,
            lhs_vec_param, rhs_vec_param,
            output_element_type, output_shape_expr,
            ntt_op_str)
        code.extend([f"    {line}" for line in golden_output_code])
        # cast_mode = 2 if datatype.cpp_type in types_to_cast else 0
        # set cast mode for back to ntt function
       


        compare_code = self.generate_compare(
            ntt_output_var_name = "ntt_output",
            golden_var_name = golden_var_name,
            ulp_tolerances = self._get_ulp_tolerance(ntt_op_str, datatype)

        )

        code.extend([f"    {line}" for line in compare_code])


        return "\n".join(code)

    def _generate_pow_test_case_pair(
            self, lhs_datatype, rhs_datatype,
            lhs_is_dynamic_shape, rhs_is_dynamic_shape,
            lhs_dims_spec, rhs_dims_spec,
            lhs_vector_rank, rhs_vector_rank,
            lhs_continuity, rhs_continuity,
            ntt_op_str):
        
        test_cases = []
        
        if lhs_datatype.cpp_type in self.integer_types:
            # Case 1: integer types - rhs is non-negative integer
            pow_ranges = self.ALL_POW_OPRANDS.get(rhs_datatype.cpp_type)
            lhs_datatype = lhs_datatype._replace(
                min_val=pow_ranges["lhs_min"],
                max_val=pow_ranges["lhs_max"]
            )
            rhs_datatype = rhs_datatype._replace(
                integer_only=True,
                min_val=pow_ranges["rhs_min"],
                max_val=pow_ranges["rhs_max"]
            )
            test_code = self.generate_test_case(
                lhs_datatype, rhs_datatype,
                lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                lhs_dims_spec, rhs_dims_spec,
                lhs_vector_rank, rhs_vector_rank,
                lhs_continuity, rhs_continuity,
                ntt_op_str
            )
            test_cases.append(test_code)
        else:
            # Case 2.1: floating point types - rhs as integer
            pow_ranges = self.ALL_POW_OPRANDS.get(lhs_datatype.cpp_type)
            lhs_datatype = lhs_datatype._replace( 
                integer_only=False,
                min_val=pow_ranges["lhs_min"],
                max_val=pow_ranges["lhs_max"]
            )
            rhs_datatype= rhs_datatype._replace(
                integer_only=True,
                min_val=pow_ranges["rhs_min"],
                max_val=pow_ranges["rhs_max"]
            )
            test_code1 = self.generate_test_case(
                lhs_datatype, rhs_datatype,
                lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                lhs_dims_spec, rhs_dims_spec,
                lhs_vector_rank, rhs_vector_rank,
                lhs_continuity, rhs_continuity,
                ntt_op_str, "rhs_int"
            )
            test_cases.append(test_code1)
            zero_val_map = {
                "bfloat16": "0.0_bf16",
                "half": "half(0.0)",
                "float_e4m3_t": "float_e4m3_t(0.0f)",
                "float_e5m2_t": "float_e5m2_t(0.0f)",
            }
            # Case 2.2: lhs is positive - rhs as float
            lhs_datatype = lhs_datatype._replace(
                min_val = zero_val_map.get(lhs_datatype.cpp_type, "0.0")
            )
            rhs_datatype = rhs_datatype._replace(
                integer_only=False
            )
            
            test_code2 = self.generate_test_case(
                lhs_datatype, rhs_datatype,
                lhs_is_dynamic_shape, rhs_is_dynamic_shape,
                lhs_dims_spec, rhs_dims_spec,
                lhs_vector_rank, rhs_vector_rank,
                lhs_continuity, rhs_continuity,
                ntt_op_str, "rhs_float"
            )
            test_cases.append(test_code2)
        
        return "\n".join(test_cases)


    def _get_param_combinations(self, op_str):
        is_dynamic_options = [False, True]
        is_view_options = [False, True]
        vector_rank_options = [0, 1, 2]  # 0: tensor, 1: 1d vector, etc. Keep it simple for now

        # Choose appropriate dims_specs based on op_str
        dims_specs_to_use = self.dims_specs_options.get(op_str, self.dims_specs_options["default"])
        
        param_combinations_exhaustive = itertools.product(
            is_dynamic_options,          # lhs_is_dynamic_shape 2
            is_dynamic_options,          # rhs_is_dynamic_shape 2
            dims_specs_to_use,           # (lhs_dims_spec, rhs_dims_spec) 6
            vector_rank_options,         # lhs_vector_rank 3
            vector_rank_options,         # rhs_vector_rank 3
            self.simple_continuities,         # lhs_continuity
            self.simple_continuities          # rhs_continuity
        )
        param_combinations_simplified = itertools.product(
            is_dynamic_options,         # lhs_is_dynamic_shape 2
            is_dynamic_options,         # rhs_is_dynamic_shape 2
            [([2, 3, 16, 16], [2, 3, 16, 16])], # (lhs_dims_spec, rhs_dims_spec)
            vector_rank_options,        # lhs_vector_rank 3
            vector_rank_options,        # rhs_vector_rank  3
            [Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None)],  # lhs_continuity
            [Continuity(is_contiguous=True, non_contiguous_dim=None, big_tensor_op=None)]
        )
        if op_str in self.op_str_map_exhaustive:
            return param_combinations_exhaustive
        else:
            return param_combinations_simplified


    def generate_all_tests_for_type(self, datatype, op_str):
        code = []
        
        code.append(self.generate_header())

        # Generate custom ORT functions if needed
        if op_str in self.ort_custom_function:
            custom_op_func = self._generate_ort_custom_op(datatype, op_str)
            code.append(custom_op_func)

        param_combinations = self._get_param_combinations(op_str)

        for lhs_is_dynamic, rhs_is_dynamic, (lhs_shape, rhs_shape), lhs_vec_rank, rhs_vec_rank, lhs_continuity, rhs_continuity in param_combinations:
            # Skip invalid combinations if any in the future
            # one element but not contiguous
            if not lhs_continuity.is_contiguous and (lhs_shape == [1]):
                continue
            if rhs_shape == [1] and not rhs_continuity.is_contiguous:
                continue

            # set non_contiguous_dim for 1 dimension tensor
            if not lhs_continuity.is_contiguous and lhs_shape == [16]:
                lhs_continuity = lhs_continuity._replace(non_contiguous_dim=0)
            if not rhs_continuity.is_contiguous and rhs_shape == [16]:
                rhs_continuity = rhs_continuity._replace(non_contiguous_dim=0)
            
            # Filter vector rank combinations for inner_product
            if op_str == "inner_product" or op_str == "outer_product":
                # Only allow: scalar x scalar, or 1D vector x 1D vector
                if not ((lhs_vec_rank == 0 and rhs_vec_rank == 0) or (lhs_vec_rank == 1 and rhs_vec_rank == 1)):
                    continue
            
            if(op_str == "pow"):
                # 1. lhs is neg or pos, rhs is int
                # 2. lhs is pos, rhs is float
                test_code = self._generate_pow_test_case_pair(
                    datatype, datatype, lhs_is_dynamic_shape=lhs_is_dynamic,
                    rhs_is_dynamic_shape=rhs_is_dynamic, lhs_dims_spec=lhs_shape,
                    rhs_dims_spec=rhs_shape, lhs_vector_rank=lhs_vec_rank,
                    rhs_vector_rank=rhs_vec_rank, lhs_continuity=lhs_continuity,
                    rhs_continuity=rhs_continuity, ntt_op_str=op_str
                )
                code.append(test_code)
            else:
                test_code = self.generate_test_case(
                    datatype, datatype, lhs_is_dynamic_shape=lhs_is_dynamic,
                    rhs_is_dynamic_shape=rhs_is_dynamic, lhs_dims_spec=lhs_shape,
                    rhs_dims_spec=rhs_shape, lhs_vector_rank=lhs_vec_rank,
                    rhs_vector_rank=rhs_vec_rank, lhs_continuity=lhs_continuity,
                    rhs_continuity=rhs_continuity, ntt_op_str=op_str
                )
                code.append(test_code)

        code.append(self.generate_footer())
        return "\n".join(code)

def generate_tests_for_op(op_str, generator):
    for datatype in ALL_DATATYPES:
        if datatype.cpp_type == "bool":
            continue
        if op_str == "ceil_div" and (datatype.cpp_type not in generator.integer_types):
            # Skip ceil_div for non-integer types, as it is only supported for integers
            continue

        test_code = generator.generate_all_tests_for_type(datatype, op_str)
        filename = f"test_ntt_binary_{datatype.name_suffix.lower()}_{op_str}_generated.cpp"
        output_filepath = os.path.join(generated_directory, filename)

        with open(output_filepath, "w") as f:
            f.write(test_code)
        
        print(f"Test file generated: {output_filepath}")
        generated_filenames.append(filename)
    

if __name__ == "__main__":
    generator = BinaryTestGenerator()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (ctest) and then the generated subdirectory
    ctest_directory = os.path.dirname(script_directory)
    generated_directory = os.path.join(ctest_directory, "generated")
    
    # Ensure generated directory exists
    os.makedirs(generated_directory, exist_ok=True)
    
    generated_filenames = []  # collect all generated file names

    # for datatype in ALL_DATATYPES:
    #     test_code = generator.generate_all_tests_for_type(datatype)
    #     filename = f"test_ntt_binary_{datatype.name_suffix.lower()}_generated.cpp"
    #     output_filepath = os.path.join(generated_directory, filename)

    #     with open(output_filepath, "w") as f:
    #         f.write(test_code)
        
    #     print(f"Test file generated: {output_filepath}")
    #     generated_filenames.append(filename)

    for op_str in (generator.op_str_map_exhaustive.keys() | generator.op_str_map_simplified.keys()):
        generate_tests_for_op(op_str, generator)
    # Generate cmake list file in the generated directory
    generate_cmake_list(generated_directory, generated_filenames, "generated_binary_tests.cmake", "GENERATED_BINARY_TEST_SOURCES")