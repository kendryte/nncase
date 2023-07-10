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

class ConstantOfShapeTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape] = GetParam();

        const int size = 768;
        int32_t array[size];

        for (int32_t &i : array) {
            i = 1;
        }

        expected =
            hrt::create(dt_int32, shape,
                        {reinterpret_cast<gsl::byte *>(array), sizeof(array)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(ConstantOfShape, ConstantOfShapeTest,
                         testing::Combine(testing::Values(dt_int32),
                                          testing::Values(dims_t{1, 3, 16,
                                                                 16})));

TEST_P(ConstantOfShapeTest, constant_of_shape) {
    //    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    //    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

    //    // expected
    //    auto output_ort = ortki_Add(l_ort, r_ort);
    //    size_t size = 0;
    //    void *ptr_ort = tensor_buffer(output_ort, &size);
    //    dims_t shape(tensor_rank(output_ort));
    //    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    //    auto expected = hrt::create(lhs.datatype(), shape,
    //                                {reinterpret_cast<gsl::byte *>(ptr_ort),
    //                                size}, true,
    //                                host_runtime_tensor::pool_cpu_only)
    //                        .expect("create tensor failed");

    // actual
    int64_t shape1[] = {1, 3, 16, 16};
    auto shape_ptr =
        hrt::create(dt_int64, {4},
                    {reinterpret_cast<gsl::byte *>(shape1), sizeof(shape1)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    int32_t value[] = {1};
    auto value_ptr =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(value), sizeof(value)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output =
        kernels::stackvm::constant_of_shape(shape_ptr.impl(), value_ptr.impl())
            .expect("constant_of_shape failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        print_runtime_tensor(actual);
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}