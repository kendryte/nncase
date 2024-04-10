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
#include "kernel_test.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <ortki/operators.h>

#define TEST_CASE_NAME "test_pad"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class PadTest : public KernelTest,
                public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");
        mode_str = GetString("mode");
        mode = Str2Mode(mode_str, str_2_padmode);
        padding = GetDataArray("padding");
        padding_nncaseformat = ToNncaseFormat(padding);
        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
        value = hrt::create(typecode, {1}, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(value);
    }
    template <typename T>
    T Str2Mode(std::string type, std::map<std::string, T> &str_2_mode) {
        std::cout << type << std::endl;
        if (str_2_mode.find(type) != str_2_mode.end()) {
            return str_2_mode[type];
        } else {
            // should not be here
            return static_cast<T>(0);
        }
    }

    std::vector<int64_t> ToNncaseFormat(std::vector<int64_t> &padding_vector) {
        // int64_t before_pad;
        // int64_t after_pad;
        std::vector<int64_t> padding_nncase;
        for (size_t i = 0; i < padding_vector.size() / 2; ++i) {
            padding_nncase.push_back(padding_vector[i]);
            padding_nncase.push_back(
                padding_vector[i + padding_vector.size() / 2]);
        }
        return padding_nncase;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor value;
    runtime::stackvm::pad_mode_t mode;
    std::string mode_str;
    std::vector<int64_t> padding;
    std::vector<int64_t> padding_nncaseformat;
    std::map<std::string, runtime::stackvm::pad_mode_t> str_2_padmode = {
        {"constant", runtime::stackvm::pad_mode_t::constant},
        {"reflect", runtime::stackvm::pad_mode_t::reflect},
        {"edge", runtime::stackvm::pad_mode_t::edge}};
};

INSTANTIATE_TEST_SUITE_P(Pad, PadTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(PadTest, Pad) {

    // expected
    size_t size = 0;
    // int64_t pad_ptr[] = {1, 0, 0, 0, 0, 0, 0, 0};
    auto pad = hrt::create(dt_int64, {padding.size()},
                           {reinterpret_cast<std::byte *>(padding.data()),
                            sizeof(padding[0]) * padding.size()},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    std::vector<int64_t> axis_v(padding.size() / 2);
    std::iota(axis_v.begin(), axis_v.end(), 0);
    auto axis = hrt::create(dt_int64, {axis_v.size()},
                            {reinterpret_cast<std::byte *>(axis_v.data()),
                             sizeof(axis_v[0]) * axis_v.size()},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

    auto l_ort = runtime_tensor_2_ort_tensor(input);
    auto pad_ort = runtime_tensor_2_ort_tensor(pad);
    auto value_ort = runtime_tensor_2_ort_tensor(value);
    auto axis_ort = runtime_tensor_2_ort_tensor(axis);
    auto output_ort =
        ortki_Pad(l_ort, pad_ort, value_ort, axis_ort, mode_str.c_str());
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    pad = hrt::create(
              dt_int64, {padding_nncaseformat.size()},
              {reinterpret_cast<std::byte *>(padding_nncaseformat.data()),
               sizeof(padding_nncaseformat[0]) * padding_nncaseformat.size()},
              true, host_runtime_tensor::pool_cpu_only)
              .expect("create tensor failed");
    auto output =
        kernels::stackvm::pad(mode, input.impl(), pad.impl(), value.impl())
            .expect("pad failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(lhs_type, j)
    FOR_LOOP(mode, k)
    FOR_LOOP(padding, l)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(mode, k)
    SPLIT_ELEMENT(padding, l)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}