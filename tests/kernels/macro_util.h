/* Copyright 2019-2021 Canaan Inc.
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
#define ORTKI_COMPUTE(op, ...) op(__VA_ARGS__)
#define ORTKI_OP_1(op, ...) ORTKI_COMPUTE(op, __VA_ARGS__)
#define ORTKI_OP_2(op_a, op_b, ...)                                            \
    ORTKI_COMPUTE(op_a, ORTKI_COMPUTE(op_b, __VA_ARGS__))
#define ORTKI_OP(num, ...) ORTKI_OP_##num(__VA_ARGS__)

#define READY_INPUT(FORMAT) READY_INPUT_##FORMAT()

#define READY_INPUT_NORMAL()                                                   \
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);                             \
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

#define READY_INPUT_VEC()                                                      \
    OrtKITensor *orts[2];                                                      \
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);                             \
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);                             \
    orts[0] = l_ort;                                                           \
    orts[1] = r_ort;                                                           \
    // auto input_size = sizeof(orts) / sizeof(orts[0]);

#define READY_INPUT_NORMAL_ARGS_1()                                            \
    auto a_ort = runtime_tensor_2_ort_tensor(a);                               \
    auto b_ort = runtime_tensor_2_vector_type(b);                              \
    auto c_ort = runtime_tensor_2_vector_type(c);                              \
    auto d_ort = runtime_tensor_2_vector_type(d);

#define GET_EXPECT(ortop_num, ...)                                             \
    auto output_ort = ORTKI_OP(ortop_num, __VA_ARGS__);                        \
    size_t size = 0;                                                           \
    void *ptr_ort = tensor_buffer(output_ort, &size);                          \
    dims_t shape(tensor_rank(output_ort));                                     \
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));       \
    auto expected =                                                            \
        hrt::create(dt_boolean, shape,                                         \
                    {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,      \
                    host_runtime_tensor::pool_cpu_only)                        \
            .expect("create tensor failed");

#define GET_ACTUAL(op_fn, op_name)                                             \
    auto output = op_fn(op_name, lhs.impl(), rhs.impl())                       \
                      .expect(std::string(#op_fn).append(" failed"));          \
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

#define GET_ACTUAL_4(op_fn, op_name)                                           \
    auto output = op_fn(op_name, a.impl(), b.impl(), c.impl(), d.impl())       \
                      .expect(std::string(#op_fn).append(" failed"));          \
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

#define CHECK_RESULT()                                                         \
    bool result = is_same_tensor(expected, actual) ||                          \
                  cosine_similarity_tensor(expected, actual);                  \
    if (!result) {                                                             \
        print_runtime_tensor(actual);                                          \
        print_runtime_tensor(expected);                                        \
    }                                                                          \
    EXPECT_TRUE(result);

#define NNCASE_TEST_BODY(test_class, test_name, op_fn, sub_op_name,            \
                         ortki_op_num, format, ...)                            \
    TEST_P(test_class, test_name) {                                            \
        READY_INPUT(format)                                                    \
        GET_EXPECT(ortki_op_num, __VA_ARGS__)                                  \
        GET_ACTUAL(op_fn, sub_op_name)                                         \
        CHECK_RESULT()                                                         \
    }

#define NNCASE_TEST_BODY_ARGS_4(test_class, test_name, op_fn, sub_op_name,     \
                                ortki_op_num, format, ...)                     \
    TEST_P(test_class, test_name) {                                            \
        READY_INPUT_NORMAL_ARGS_1()                                            \
        GET_EXPECT(ortki_op_num, __VA_ARGS__)                                  \
        GET_ACTUAL_4(op_fn, sub_op_name)                                       \
        CHECK_RESULT()                                                         \
    }

#define NNCASE_TEST_CLASS(class_name)                                          \
    class class_name : public KernelTest,                                      \
                       public ::testing::TestWithParam<                        \
                           std::tuple<nncase::typecode_t, dims_t, dims_t>> {   \
      public:                                                                  \
        void SetUp() override {                                                \
            auto &&[typecode, l_shape, r_shape] = GetParam();                  \
            lhs = hrt::create(typecode, l_shape,                               \
                              host_runtime_tensor::pool_cpu_only)              \
                      .expect("create tensor failed");                         \
            rhs = hrt::create(typecode, r_shape,                               \
                              host_runtime_tensor::pool_cpu_only)              \
                      .expect("create tensor failed");                         \
            init_tensor(lhs);                                                  \
            init_tensor(rhs);                                                  \
        }                                                                      \
        void TearDown() override {}                                            \
                                                                               \
      protected:                                                               \
        runtime_tensor lhs;                                                    \
        runtime_tensor rhs;                                                    \
    };

#define NNCASE_TEST_CLASS_ARGS_4(class_name)                                   \
    class class_name                                                           \
        : public KernelTest,                                                   \
          public ::testing::TestWithParam<                                     \
              std::tuple<typecode_t, typecode_t, typecode_t, typecode_t,       \
                         dims_t, dims_t, dims_t, dims_t>> {                    \
      public:                                                                  \
        void SetUp() override {                                                \
            auto &&[typecode_a, typecode_b, typecode_c, typecode_d, shape_a,   \
                    shape_b, shape_c, shape_d] = GetParam();                   \
            a = hrt::create(typecode_a, shape_a,                               \
                            host_runtime_tensor::pool_cpu_only)                \
                    .expect("create tensor failed");                           \
            b = hrt::create(typecode_b, shape_b,                               \
                            host_runtime_tensor::pool_cpu_only)                \
                    .expect("create tensor failed");                           \
            c = hrt::create(typecode_c, shape_c,                               \
                            host_runtime_tensor::pool_cpu_only)                \
                    .expect("create tensor failed");                           \
            d = hrt::create(typecode_d, shape_d,                               \
                            host_runtime_tensor::pool_cpu_only)                \
                    .expect("create tensor failed");                           \
            init_tensor(a);                                                    \
            init_tensor(b);                                                    \
            init_tensor(c);                                                    \
            init_tensor(d);                                                    \
        }                                                                      \
        void TearDown() override {}                                            \
                                                                               \
      protected:                                                               \
        runtime_tensor a;                                                      \
        runtime_tensor b;                                                      \
        runtime_tensor c;                                                      \
        runtime_tensor d;                                                      \
    };

#define NNCASE_TEST_GENERATOR_VALUE(...) testing::Values(__VA_ARGS__)
#define NNCASE_TEST_VALUE(...) testing::Values(__VA_ARGS__)

#define NNCASE_TEST_GENERATOR_TYPE_3(typea, typeb, typec, ...)                 \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea, typeb, typec),         \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__),                 \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__))

#define NNCASE_TEST_GENERATOR_TYPE_2(typea, typeb, ...)                        \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea, typeb),                \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__),                 \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__))

#define NNCASE_TEST_GENERATOR_8(a, b, c, d, e, f, g, h)                        \
    testing::Combine(NNCASE_TEST_VALUE(a), NNCASE_TEST_VALUE(b),               \
                     NNCASE_TEST_VALUE(c), NNCASE_TEST_VALUE(d),               \
                     NNCASE_TEST_VALUE(e), NNCASE_TEST_VALUE(f),               \
                     NNCASE_TEST_VALUE(g), NNCASE_TEST_VALUE(h))

#define NNCASE_TESTSUITE_INIT(test_class, test_name, dtype_nums, ...)          \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        test_name, test_class,                                                 \
        NNCASE_TEST_GENERATOR_TYPE_##dtype_nums(__VA_ARGS__));

#define NNCASE_TESTSUITE_INIT_COMBINE(test_class, test_name, ...)              \
    INSTANTIATE_TEST_SUITE_P(test_name, test_class,                            \
                             NNCASE_TEST_GENERATOR_8(__VA_ARGS__));

#define GET_DEFAULT_TEST_SHAPE()                                               \
    dims_t{1, 3, 16, 16}, dims_t{3, 16, 16}, dims_t{3, 16, 1}, dims_t{16, 16}, \
        dims_t{16, 1}, dims_t{1, 16, 1}, dims_t{16}, dims_t{1}, dims_t {}

#define GET_DEFAULT_TEST_VECTOR_SHAPE()                                        \
    dims_t { 4 }

#define GET_DEFAULT_TEST_SCALAR_SHAPE()                                        \
    dims_t { 1 }

#define GET_DEFAULT_TEST_TYPE_3() dt_float32, dt_int32, dt_int64
#define GET_DEFAULT_TEST_INTTYPE() dt_int32, dt_int64

#define NNCASE_CONDITION(condition)                                            \
    case dt_##condition:                                                       \
        arr = new condition##_t[tensor.shape().size()];                        \
        break;

#define NNCASE_CONDITION_GET(condition)                                        \
    case dt_##condition:                                                       \
        arr[index] =                                                           \
            static_cast<condition##_t>(get<condition##_t>(tensor, index));     \
        break;
