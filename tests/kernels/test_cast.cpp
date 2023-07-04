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

        expected = hrt::create(typecode_output, l_shape,
                             host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor expected;
};

// todo There is a issue with converting fixed-point numbers to floating-point numbers.
INSTANTIATE_TEST_SUITE_P(
    Cast, CastTest,
    testing::Combine(
        testing::Values(dt_float32, dt_float16, dt_int32, dt_int8, dt_int16),
        testing::Values(dt_int16, dt_int8, dt_int32, dt_int64),
        testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3, 8, 8},
                        dims_t{1, 3, 1})));

TEST_P(CastTest, cast) {
    // actual
    auto output =
        kernels::stackvm::cast(
            expected.datatype(), runtime::stackvm::cast_mode_t::kdefault, input.impl())
            .expect("cast failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // expected
    cast_copy_tensor(input, expected);

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual) ||
                cosine_similarity_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}