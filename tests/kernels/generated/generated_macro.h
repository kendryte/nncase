/* Copyright 2019-2023 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define NNCASE_TEST_CLASS_ARGS_2_ATTR_0(class_name, init_f0, init_f1)          \
    class class_name                                                           \
        : public KernelTest,                                                   \
          public ::testing::TestWithParam<                                     \
              std::tuple<typecode_t, typecode_t, dims_t, dims_t>> {            \
      public:                                                                  \
        void SetUp() override {                                                \
            auto &&[typecode_0, typecode_1, shape_0, shape_1] = GetParam();    \
            _typecode_0 = typecode_0;                                          \
            a0 = hrt::create(typecode_0, shape_0,                              \
                             host_runtime_tensor::pool_cpu_only)               \
                     .expect("create tensor failed");                          \
            INIT_TENSOR(a0, init_f0);                                          \
            a1 = hrt::create(typecode_1, shape_1,                              \
                             host_runtime_tensor::pool_cpu_only)               \
                     .expect("create tensor failed");                          \
            INIT_TENSOR(a1, init_f1);                                          \
        }                                                                      \
        void TearDown() override { CLEAR_SUBCASE() }                                            \
                                                                               \
      protected:                                                               \
        runtime_tensor a0;                                                     \
        typecode_t _typecode_0;                                                \
        runtime_tensor a1;                                                     \
        typecode_t _typecode_1;                                                \
    };
#define READY_INPUT_ARGS_2()                                                   \
    auto a0_ort = runtime_tensor_2_ort_tensor(a0);                             \
    auto a1_ort = runtime_tensor_2_ort_tensor(a1);
#define GET_ACTUAL_ARGS_2_ATTR_0(op_fn, op_name)                               \
    auto output = op_fn(op_name, a0.impl(), a1.impl())                         \
                      .expect(std::string(#op_fn).append(" failed"));          \
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));
#define GET_EXPECT_ARGS_2_ATTR_0(op)                                           \
    auto output_ort = op(a0_ort, a1_ort);                                      \
    size_t size = 0;                                                           \
    void *ptr_ort = tensor_buffer(output_ort, &size);                          \
    dims_t shape(tensor_rank(output_ort));                                     \
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
#define NNCASE_TEST_BODY_ARGS_2_ATTR_0(test_class, test_name, op_fn,           \
                                       sub_op_name, ort_op, out_compare_type)  \
    TEST_P(test_class, test_name) {                                            \
        READY_INPUT_ARGS_2()                                                   \
        GET_EXPECT_ARGS_2_ATTR_0(ort_op)                                       \
        CONVERT_EXPECT_TO_RT(out_compare_type)                                 \
        GET_ACTUAL_ARGS_2_ATTR_0(op_fn, sub_op_name)                           \
        CHECK_RESULT()                                                         \
    }
#define NNCASE_TESTSUITE_INIT_ARGS_2(test_class, test_name, type0, type1,      \
                                     shape0, shape1)                           \
    INSTANTIATE_TEST_SUITE_P(test_name, test_class,                            \
                             testing::Combine(type0, type1, shape0, shape1));
