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
        hrt::create(_typecode, shape,                                          \
                    {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,      \
                    host_runtime_tensor::pool_cpu_only)                        \
            .expect("create tensor failed");

#define GET_EXPECT_BOOL(ortop_num, ...)                                        \
    auto output_ort = ORTKI_OP(ortop_num, __VA_ARGS__);                        \
    size_t size = 0;                                                           \
    void *ptr_ort = tensor_buffer(output_ort, &size);                          \
    dims_t shape(tensor_rank(output_ort));                                     \
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));       \
    auto expected =                                                            \
        hrt::create(dt_boolean, shape,                                         \
                    {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,      \
                    host_runtime_tensor::pool_cpu_only)                        \
            .expect("create expected tensor failed");

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
        std::cout << "actual ";                                                \
        print_runtime_tensor(actual);                                          \
        std::cout << "expected ";                                              \
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

#define NNCASE_TEST_BODY_BOOL(test_class, test_name, op_fn, sub_op_name,       \
                              ortki_op_num, format, ...)                       \
    TEST_P(test_class, test_name) {                                            \
        READY_INPUT(format)                                                    \
        GET_EXPECT_BOOL(ortki_op_num, __VA_ARGS__)                             \
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
            _typecode = typecode;                                              \
            lhs = hrt::create(typecode, l_shape,                               \
                              host_runtime_tensor::pool_cpu_only)              \
                      .expect("create lhs tensor failed");                     \
            rhs = hrt::create(typecode, r_shape,                               \
                              host_runtime_tensor::pool_cpu_only)              \
                      .expect("create rhs tensor failed");                     \
            init_tensor(lhs);                                                  \
            init_tensor(rhs);                                                  \
        }                                                                      \
        void TearDown() override {}                                            \
                                                                               \
      protected:                                                               \
        runtime_tensor lhs;                                                    \
        runtime_tensor rhs;                                                    \
        typecode_t _typecode;                                                  \
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

#define NNCASE_TEST_GENERATOR_TYPE_1(typea, ...)                               \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea),                       \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__),                 \
                     NNCASE_TEST_GENERATOR_VALUE(__VA_ARGS__))

#define NNCASE_TEST_GENERATOR_TYPE_11(typea, ...)                              \
    testing::Combine(NNCASE_TEST_GENERATOR_VALUE(typea),                       \
                     NNCASE_TEST_GENERATOR_VALUE(typea),                       \
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

#define GET_DEFAULT_TEST_SHAPE_()                                              \
    testing::Values(dims_t{1, 3, 16, 16}, dims_t{3, 16, 16}, dims_t{3, 16, 1}, \
                    dims_t{16, 16}, dims_t{16, 1}, dims_t{1, 16, 1},           \
                    dims_t{16}, dims_t{1}, dims_t{})

#define GET_DEFAULT_TEST_TYPE_() testing::Values(dt_int32)

#define GET_DEFAULT_TEST_BOOL_TYPE_() testing::Values(dt_boolean)

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

#define CreateRtFromAttr_SCALAR(attr_rt_type, attr_shape, attr)                \
    hrt::create(attr_rt_type, attr_shape,                                      \
                {reinterpret_cast<gsl::byte *>(&attr), _msize(attr)}, true,    \
                host_runtime_tensor::pool_cpu_only)                            \
        .expect("create tensor failed");

#define CreateRtFromAttr_ARRAYONEDIM(attr_rt_type, attr_shape, attr)           \
    hrt::create(attr_rt_type, attr_shape,                                      \
                {reinterpret_cast<gsl::byte *>(attr), _msize(attr)}, true,     \
                host_runtime_tensor::pool_cpu_only)                            \
        .expect("create tensor failed");

#define DECLARE_ATTR(attr, attr_type_id, attr_type)                            \
    DECLARE_ATTR_##attr_type_id(attr, attr_type)

#define DECLARE_ATTR_SCALAR(attr, attr_type) attr_type attr;
#define DECLARE_ATTR_ARRAYONEDIM(attr, attr_type) attr_type *attr;
#define DECLARE_ATTR_ARRAYTWODIM(attr, attr_type) attr_type **attr;

#define SWITCH_INIT_MODE(type, dis1, dis2, dis3, dis4)                         \
    switch (mode) {                                                            \
    case RANDOM:                                                               \
        get<type>(tensor, index) = static_cast<type>(dis1(gen));               \
        break;                                                                 \
    case NOZERO:                                                               \
        get<type>(tensor, index) = static_cast<type>(dis2(gen));               \
        break;                                                                 \
    case NONEG:                                                                \
        get<type>(tensor, index) = static_cast<type>(dis3(gen));               \
        break;                                                                 \
    case NOPOS:                                                                \
        get<type>(tensor, index) = static_cast<type>(dis4(gen));               \
        break;                                                                 \
    default: {                                                                 \
        get<type>(tensor, index) = static_cast<type>(dis1(gen));               \
        break;                                                                 \
    }                                                                          \
    }

#define INIT_TENSOR(a, init_mode) InitTensor(a, init_mode)

#define INIT_ATTRIBUTE(attr_type_id, attr_type, attr_shape, attr_initf)        \
    InitAttribute##attr_type_id<attr_type>(attr_shape, attr_initf)

#define CreateRtFromAttr(attr_type_id, attr_rt_type, attr_shape, attr)         \
    CreateRtFromAttr_##attr_type_id(attr_rt_type, attr_shape, attr)

#define CONVERT_EXPECT_TO_RT(type)                                             \
    auto expected =                                                            \
        hrt::create(type, shape,                                               \
                    {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,      \
                    host_runtime_tensor::pool_cpu_only)                        \
            .expect("create expected tensor failed");

#define MAX_CASE_NUM 10000
#define ENDFIX ".json"
#define PARENT_DIR_1 "../../../tests/kernels/"
#define PARENT_DIR_2 "../../../../tests/kernels/"

#define SPLIT_ELEMENT(key, idx)                                                \
    rapidjson::Value copiedArray##key(rapidjson::kArrayType);                  \
    copiedArray##key.CopyFrom(key[idx], write_doc.GetAllocator());             \
    write_doc.AddMember(Value(#key, write_doc.GetAllocator()),                 \
                        copiedArray##key, write_doc.GetAllocator());

#define FOR_LOOP(key, idx)                                                     \
    assert(document[#key].IsArray());                                          \
    Value &key = document[#key];                                               \
    for (SizeType idx = 0; idx < key.Size(); ++idx) {

#define FOR_LOOP_END() }

#define FILE_NAME_GEN(PARENT_DIR, name)                                        \
    std::string(PARENT_DIR) + std::string(name) + std::string(ENDFIX)

#define FILE_NAME_GEN_SUBCASE(case_name, filename, idx)                        \
    std::string(case_name) + "_" + std::string(filename) + "_" +               \
        std::to_string(idx) + std::string(ENDFIX)

#define READY_TEST_CASE_GENERATE()                                             \
    std::string content;                                                       \
    auto filename1 = FILE_NAME_GEN(PARENT_DIR_1, TEST_CASE_NAME);              \
    std::ifstream file1(filename1);                                            \
    if (file1.fail()) {                                                        \
        file1.close();                                                         \
        auto filename2 = FILE_NAME_GEN(PARENT_DIR_2, TEST_CASE_NAME);          \
        std::ifstream file2(filename2);                                        \
        if (file2.fail()) {                                                    \
            file2.close();                                                     \
            std::cout << "File does not exist: " << filename2 << std::endl;    \
        } else {                                                               \
            content = KernelTest::ReadFromJsonFile(file2);                     \
            std::cout << "File exists: " << filename2 << std::endl;            \
        }                                                                      \
    } else {                                                                   \
        content = KernelTest::ReadFromJsonFile(file1);                         \
        std::cout << "File exists: " << filename1 << std::endl;                \
    }                                                                          \
    Document document;                                                         \
    KernelTest::ParseJson(document, content);                                  \
    unsigned case_num = 0;                                                     \
    Document write_doc;                                                        \
    write_doc.SetObject();

#define WRITE_SUB_CASE()                                                       \
    std::ofstream ofs(FILE_NAME_GEN_SUBCASE(                                   \
        TEST_CASE_NAME, KernelTest::GetFileNameFromMacro(__FILE__),            \
        case_num));                                                            \
    OStreamWrapper osw(ofs);                                                   \
    Writer<OStreamWrapper> writer(osw);                                        \
    write_doc.Accept(writer);                                                  \
    case_num++;                                                                \
    write_doc.RemoveAllMembers();

#define READY_SUBCASE()                                                        \
    auto &&[idx] = GetParam();                                                 \
    auto filename = FILE_NAME_GEN_SUBCASE(                                     \
        TEST_CASE_NAME, KernelTest::GetFileNameFromMacro(__FILE__), idx);      \
    std::ifstream file(filename);                                              \
    if (file.is_open()) {                                                      \
        std::cout << "Open file: " << filename << std::endl;                   \
        ParseJson(ReadFromJsonFile(file));                                     \
    } else {                                                                   \
        file.close();                                                          \
        GTEST_SKIP();                                                          \
    }

#define CLEAR_SUBCASE()                                                        \
    auto &&[idx] = GetParam();                                                 \
    auto filename = FILE_NAME_GEN_SUBCASE(                                     \
        TEST_CASE_NAME, KernelTest::GetFileNameFromMacro(__FILE__), idx);      \
    if (std::remove(filename.c_str()) == 0) {                                  \
        printf("File deleted successfully: %s\n", filename.c_str());           \
    }
