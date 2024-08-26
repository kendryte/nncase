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

#define TEST_CASE_NAME "test_prelu"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;
using slope_t = itlib::small_vector<float, 4>;

class PreluTest : public KernelTest,
                  public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");
        auto slope_value = GetSlopeArray("slope");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        if (slope_value.size() == 1 ||
            slope_value.size() == l_shape[l_shape.size() - 1]) {
            slope = slope_value;
        } else {
            slope = slope_t{0.1};
        }

        size_t slope_size = slope.size();
        float *slope_array = (float *)malloc(slope_size * sizeof(float));
        std::copy(slope.begin(), slope.end(), slope_array);
        slope_tensor = hrt::create(dt_float32, {slope_size},
                                   {reinterpret_cast<std::byte *>(slope_array),
                                    slope_size * sizeof(float)},
                                   true, host_runtime_tensor::pool_cpu_only)
                           .expect("create tensor failed");
    }

    void TearDown() override{CLEAR_SUBCASE()}

    slope_t GetSlopeArray(const char *key) {
        assert(_document[key].is_array());
        const auto &array = _document[key];
        size_t arraySize = array.size();
        slope_t cArray(arraySize);
        for (size_t i = 0; i < arraySize; i++) {
            if (array[i].is_number_float()) {
                cArray[i] = array[i].get<float>();
            } else {
                std::cout << "Invalid JSON format. Expected unsigned float "
                             "values in the array."
                          << std::endl;
            }
        }
        return cArray;
    }

  protected:
    runtime_tensor input;
    runtime_tensor slope_tensor;
    slope_t slope;
};

INSTANTIATE_TEST_SUITE_P(Prelu, PreluTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(PreluTest, Prelu) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    runtime_tensor slope_tensor_like_input(
        kernels::stackvm::cast(input.datatype(),
                               runtime::stackvm::cast_mode_t::kdefault,
                               slope_tensor.impl())
            .expect("cast failed")
            .as<tensor>()
            .expect("as tensor failed"));
    auto slope_ort = runtime_tensor_2_ort_tensor(slope_tensor_like_input);
    auto output_ort = ortki_PRelu(l_ort, slope_ort);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::prelu(input.impl(), slope_tensor_like_input.impl())
            .expect("prelu failed");
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
    FOR_LOOP(slope, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(slope, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}