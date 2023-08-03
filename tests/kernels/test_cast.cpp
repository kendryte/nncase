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
#include "nncase/runtime/datatypes.h"
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

class CastTest : public KernelTest,
                 public ::testing::TestWithParam<
                     std::tuple<nncase::typecode_t, typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode_input, typecode_output, l_shape] = GetParam();

        input = hrt::create(typecode_input, l_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(input);

        input1 =
            hrt::create(dt_float16, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input1);

        input2 =
            hrt::create(dt_float32, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input2);

        expected = hrt::create(typecode_output, l_shape,
                               host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor input1;
    runtime_tensor input2;
    runtime_tensor expected;
};

INSTANTIATE_TEST_SUITE_P(
    cast, CastTest,
    testing::Combine(testing::Values(dt_int16, dt_int8, dt_float32, dt_uint8),
                     testing::Values(dt_int16, dt_int8, dt_float32, dt_uint8),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3, 8, 8},
                                     dims_t{1, 3, 1})));

TEST_P(CastTest, cast) {
    // actual
    auto output = kernels::stackvm::cast(
                      expected.datatype(),
                      runtime::stackvm::cast_mode_t::kdefault, input.impl())
                      .expect("cast failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // expected
    //    cast_copy_tensor(input, expected);
    auto output_ort = ortki_CastLike(runtime_tensor_2_ort_tensor(input),
                                     runtime_tensor_2_ort_tensor(actual));
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(actual.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

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

    // actual
    auto output1 =
        kernels::stackvm::cast(
            dt_float32, runtime::stackvm::cast_mode_t::kdefault, input1.impl())
            .expect("cast failed");
    runtime_tensor actual1(output1.as<tensor>().expect("as tensor failed"));

    // expected
    //    cast_copy_tensor(input, expected);
    auto output_ort1 = ortki_CastLike(runtime_tensor_2_ort_tensor(input1),
                                      runtime_tensor_2_ort_tensor(actual1));
    size_t size1 = 0;
    void *ptr_ort1 = tensor_buffer(output_ort1, &size1);
    dims_t shape1(tensor_rank(output_ort1));
    tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
    auto expected1 =
        hrt::create(actual1.datatype(), shape1,
                    {reinterpret_cast<gsl::byte *>(ptr_ort1), size1}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    bool result1 = is_same_tensor(expected1, actual1) ||
                   cosine_similarity_tensor(expected1, actual1);

    if (!result1) {
        std::cout << "actual ";
        print_runtime_tensor(actual);
        std::cout << "expected ";
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result1);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}