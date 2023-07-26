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

class GetItemTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(get_item, GetItemTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1})));

TEST_P(GetItemTest, get_item) {

    // expected
    auto expected = input;

    // actual
    int64_t index_ptr[] = {0};
    auto index = hrt::create(nncase::dt_int64, {1},
                             {reinterpret_cast<gsl::byte *>(index_ptr),
                              sizeof(index_ptr)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    int64_t shape_ort[] = {1};
    auto shape = hrt::create(dt_int64, {1},
                             {reinterpret_cast<gsl::byte *>(shape_ort),
                              sizeof(shape_ort)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    auto get_item_output =
        kernels::stackvm::get_item(input.impl(), index.impl())
            .expect("get_item failed");

    auto output = kernels::stackvm::reshape(get_item_output, shape.impl())
                      .expect("get_item failed");
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
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}