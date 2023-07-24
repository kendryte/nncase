function GenerateHeader() {
    echo  "/* Copyright 2019-2023 Canaan Inc.
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
*/" > ${generated_file}
}


function GenerateNncaseTestClassMacro() {
    echo "Generating NncaseTestClassMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."
    echo -n "#define NNCASE_TEST_CLASS_ARGS_${ARGS}_ATTR_${ATTR}(class_name, " >> ${generated_file}
    for ((i = 0; i < ARGS; i++))
    do
        echo -n "init_f${i}," >> ${generated_file}
    done

    for ((i = 0; i < ATTR; i++))
    do
        echo -n "attr_type_id${i}, attr_type${i}, attr_rt_type${i}, attr_shape${i}, attr_initf${i}," >> ${generated_file}
    done

    sed -i '$s/.$//' ${generated_file}
    echo ") \\" >> ${generated_file}

    echo -n "class class_name \\
            : public KernelTest, \\
            public ::testing::TestWithParam< \\
                std::tuple<" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
        echo -n "typecode_t," >> ${generated_file}
    done
    for ((i = 0; i < ARGS; i++))
    do
        echo -n "dims_t," >> ${generated_file}
    done
    sed -i '$s/.$//' ${generated_file}
    echo ">> {\\" >> ${generated_file}

    echo -n "public: \\
            void SetUp() override { \\
                auto &&[" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
        echo -n "typecode_${i}," >> ${generated_file}
    done
    for ((i = 0; i < ARGS; i++))
    do
        echo -n "shape_${i}," >> ${generated_file}
    done
    sed -i '$s/.$//' ${generated_file}
    echo "] = GetParam(); \\" >> ${generated_file}

    echo "_typecode_0 = typecode_0; \\" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
    echo "a${i} = hrt::create(typecode_${i}, shape_${i}, host_runtime_tensor::pool_cpu_only).expect(\"create tensor failed\"); \\
            INIT_TENSOR(a${i}, init_f${i}); \\" >> ${generated_file}
    done

    for ((i = 0; i < ATTR; i++))
    do
    echo "attr${i} = INIT_ATTRIBUTE(attr_type_id${i}, attr_type${i}, attr_shape${i}, attr_initf${i}); \\
                rt_attr${i} = CreateRtFromAttr(attr_type_id${i}, attr_rt_type${i}, attr_shape${i}, attr${i}) \\" >> ${generated_file}
    done

    echo "}                                                                      \\
            void TearDown() override {}                                            \\
        protected:                                                               \\" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
    echo "runtime_tensor a${i};                                                      \\
        typecode_t _typecode_${i}; \\" >> ${generated_file}
    done

    for ((i = 0; i < ATTR; i++))
    do
    echo "DECLARE_ATTR(attr${i}, attr_type_id${i}, attr_type${i}) \\
            runtime_tensor rt_attr${i};                                                      \\" >> ${generated_file}
    done

    echo "};" >> ${generated_file}
    echo "Finished generate NncaseTestClassMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

}

function GenerateInputMacro() {
    echo "Generating InputMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."
    echo "#define READY_INPUT_ARGS_${ARGS}() \\" >> ${generated_file}
    for ((i = 0; i < ARGS; i++))
    do
        echo "auto a${i}_ort = runtime_tensor_2_ort_tensor(a${i}); \\" >> ${generated_file}
    done
    sed -i '$s/.$//' ${generated_file}
    echo "Finished generate InputMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

}


function GenerateGetActualMacro() {
    echo "Generating GetActualMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

    echo -n "#define GET_ACTUAL_ARGS_${ARGS}_ATTR_${ATTR}(op_fn, op_name) \\
        auto output = op_fn(op_name" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
        echo -n ", a${i}.impl()" >> ${generated_file}
    done

    for ((i = 0; i < ATTR; i++))
    do
        echo -n ", rt_attr${i}.impl()" >> ${generated_file}
    done
        
    echo ").expect(std::string(#op_fn).append(\" failed\"));  \\
        runtime_tensor actual(output.as<tensor>().expect(\"as tensor failed\"));" >> ${generated_file}
    echo "Finished generate GetActualMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."
    
}

function GenerateGetExpectMacro() {
    echo "Generating GetExpectMacr for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

    echo -n "#define GET_EXPECT_ARGS_${ARGS}_ATTR_${ATTR}(op) \\
            auto output_ort = op(" >> ${generated_file}

    for ((i = 0; i < ARGS; i++))
    do
        echo -n "a${i}_ort," >> ${generated_file}
    done

    for ((i = 0; i < ATTR; i++))
    do
        echo -n "attr${i}," >> ${generated_file}
    done
    sed -i '$s/.$//' ${generated_file}
    echo ");                        \\
        size_t size = 0;                                                           \\
        void *ptr_ort = tensor_buffer(output_ort, &size);                          \\
        dims_t shape(tensor_rank(output_ort));                                     \\
        tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));" >> ${generated_file}
    echo "Finished generate GetExpectMacr for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

}

function GenerateNncaseTestBodyMacro() {
    echo "Generating NncaseTestBodyMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

    echo "#define NNCASE_TEST_BODY_ARGS_${ARGS}_ATTR_${ATTR}(test_class, test_name, op_fn, sub_op_name, ort_op, out_compare_type) \\" >> ${generated_file}
    echo "TEST_P(test_class, test_name) { \\
            READY_INPUT_ARGS_${ARGS}() \\
            GET_EXPECT_ARGS_${ARGS}_ATTR_${ATTR}(ort_op) \\
            CONVERT_EXPECT_TO_RT(out_compare_type) \\
            GET_ACTUAL_ARGS_${ARGS}_ATTR_${ATTR}(op_fn, sub_op_name)  \\
            CHECK_RESULT()  \\
        }" >> ${generated_file}
    echo "Finished generate NncaseTestBodyMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

}

function GenerateNncaseTestSuiteMacro() {
    echo "Generating NncaseTestSuiteMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

    echo -n "#define NNCASE_TESTSUITE_INIT_ARGS_${ARGS}(test_class,test_name" >> ${generated_file}
    for ((i = 0; i < ARGS; i++))
    do
        echo -n ",type${i}" >> ${generated_file}
    done
    for ((i = 0; i < ARGS; i++))
    do
        echo -n ",shape${i}" >> ${generated_file}
    done
    echo -n ") \\
    INSTANTIATE_TEST_SUITE_P(                                                  \\
        test_name, test_class,                                                 \\
        testing::Combine(" >> ${generated_file}
    for ((i = 0; i < ARGS; i++))
    do
        echo -n "type${i}," >> ${generated_file}
    done
    for ((i = 0; i < ARGS; i++))
    do
        echo -n "shape${i}," >> ${generated_file}
    done
    sed -i '$s/.$//' ${generated_file}

    echo  "));" >> ${generated_file}
    echo "Finished generate NncaseTestSuiteMacro for runtime kernel with ${ARGS} Inputs and ${ATTR} Attributes..."

}

function ParseOpConfig() {
    # echo "$1"
    OLD_IFS="$IFS" 
    IFS=','
    op_config=($1) 
    IFS="$OLD_IFS" 
    # # for s in ${arr[@]} 
    # # do 
    # #     echo "$s" 
    # # done
    # op_config=$arr
    # for s in ${op_config[@]} 
    # do 
    #     echo "$s" 
    # done
}