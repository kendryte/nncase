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

#define TEST_CASE_NAME "test_clamp"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class ClampTest : public KernelTest,
                  public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        auto value1 = GetFloatNumber("min");
        auto value2 = GetFloatNumber("max");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        min_value = value1;
        max_value = value2;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    float min_value;
    float max_value;
};

INSTANTIATE_TEST_SUITE_P(clamp, ClampTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ClampTest, clamp) {

    // expected
    float min[] = {min_value};
    auto min_tensor_float =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(min), sizeof(min)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    float max[] = {max_value};
    auto max_tensor_float =
        hrt::create(nncase::dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(max), sizeof(max)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output_ort =
        ortki_Clip(runtime_tensor_2_ort_tensor(input),
                   ortki_CastLike(runtime_tensor_2_ort_tensor(min_tensor_float),
                                  runtime_tensor_2_ort_tensor(input), 1),
                   ortki_CastLike(runtime_tensor_2_ort_tensor(max_tensor_float),
                                  runtime_tensor_2_ort_tensor(input), 1));
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    runtime_tensor min_tensor(
        kernels::stackvm::cast(input.datatype(),
                               runtime::stackvm::cast_mode_t::kdefault,
                               min_tensor_float.impl())
            .expect("cast failed")
            .as<tensor>()
            .expect("as tensor failed"));

    runtime_tensor max_tensor(
        kernels::stackvm::cast(input.datatype(),
                               runtime::stackvm::cast_mode_t::kdefault,
                               max_tensor_float.impl())
            .expect("cast failed")
            .as<tensor>()
            .expect("as tensor failed"));

    auto output = kernels::stackvm::clamp(input.impl(), min_tensor.impl(),
                                          max_tensor.impl())
                      .expect("clamp failed");
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
    FOR_LOOP(min, k)
    FOR_LOOP(max, l)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(min, k)
    SPLIT_ELEMENT(max, l)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}