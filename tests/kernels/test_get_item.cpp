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

#define TEST_CASE_NAME "test_get_item"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class GetItemTest : public KernelTest,
                    public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto typecode = GetDataType("lhs_type");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
};

INSTANTIATE_TEST_SUITE_P(get_item, GetItemTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(GetItemTest, get_item) {

    // expected
    auto expected = input;

    // actual
    int64_t index_ptr[] = {0};
    auto index = hrt::create(nncase::dt_int64, {1},
                             {reinterpret_cast<std::byte *>(index_ptr),
                              sizeof(index_ptr)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    auto get_item_output =
        kernels::stackvm::get_item(input.impl(), index.impl())
            .expect("get_item failed");

    auto output = get_item_output;
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
    FOR_LOOP(lhs_shape, j)
    FOR_LOOP(lhs_type, i)
    SPLIT_ELEMENT(lhs_shape, j)
    SPLIT_ELEMENT(lhs_type, i)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}