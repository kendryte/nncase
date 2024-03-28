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

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

#define TEST_CASE_NAME "test_uniform"

class UniformTest : public KernelTest,
                    public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode = GetDataType("lhs_type");
        auto l_shape = GetAxesArray("l_shape");
        auto shape = GetShapeArray("shape");
        auto value1 = GetFloatNumber("value1");
        auto value2 = GetFloatNumber("value2");
        auto value3 = GetFloatNumber("value3");

        high_value = value1;
        float high_array[] = {high_value};
        high = hrt::create(typecode, shape,
                           {reinterpret_cast<std::byte *>(high_array),
                            sizeof(high_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");

        low_value = value2;
        float low_array[] = {low_value};
        low = hrt::create(
                  typecode, shape,
                  {reinterpret_cast<std::byte *>(low_array), sizeof(low_array)},
                  true, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        seed_value = value3;
        float seed_array[] = {seed_value};
        seed = hrt::create(typecode, shape,
                           {reinterpret_cast<std::byte *>(seed_array),
                            sizeof(seed_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");

        shape_array = l_shape;
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor high;
    runtime_tensor low;
    runtime_tensor seed;
    axes_t shape_array;
    float high_value;
    float low_value;
    float seed_value;
};

INSTANTIATE_TEST_SUITE_P(Uniform, UniformTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(UniformTest, Uniform) {
    // expected
    std::vector<int64_t> vec(shape_array.begin(), shape_array.end());
    int64_t shape_u_array[4];
    std::copy(vec.begin(), vec.end(), shape_u_array);
    auto output_ort = ortki_RandomUniform(1, high_value, low_value, seed_value,
                                          shape_u_array, 4);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto shape_u = hrt::create(dt_int64, {4},
                               {reinterpret_cast<std::byte *>(shape_u_array),
                                sizeof(shape_u_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    auto output = kernels::stackvm::uniform(dt_float32, high.impl(), low.impl(),
                                            seed.impl(), shape_u.impl())
                      .expect("uniform failed");
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
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(shape, j)
    FOR_LOOP(l_shape, k)
    FOR_LOOP(value1, l)
    FOR_LOOP(value2, m)
    FOR_LOOP(value3, n)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(shape, j)
    SPLIT_ELEMENT(l_shape, k)
    SPLIT_ELEMENT(value1, l)
    SPLIT_ELEMENT(value2, m)
    SPLIT_ELEMENT(value3, n)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}