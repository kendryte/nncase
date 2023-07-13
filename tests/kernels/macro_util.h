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

#define NNCASE_TEST_GENERATOR_VALUE(...) testing::Values(__VA_ARGS__)

#define NNCASE_TEST_GENERATOR_TYPE_3(typea, typeb, typec, ...)                 \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea, typeb, typec),         \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__),                 \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__))

#define NNCASE_TEST_GENERATOR_TYPE_2(typea, typeb, ...)                        \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea, typeb),                \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__),                 \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__))

#define NNCASE_TESTSUITE_INIT(test_class, test_name, dtype_nums, ...)          \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        test_name, test_class,                                                 \
        NNCASE_TEST_GENERATOR_TYPE_##dtype_nums(__VA_ARGS__));
