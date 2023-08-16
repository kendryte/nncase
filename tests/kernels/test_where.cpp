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

#define TEST_CASE_NAME "test_where"

class WhereTest : public KernelTest,
                  public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode = GetDataType("lhs_type");
        auto typecode_bool = GetDataType("rhs_type");
        auto shape = GetShapeArray("i_shape");

        lhs = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);

        rhs = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(rhs);

        con = hrt::create(typecode_bool, shape,
                          host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(con);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor lhs;
    runtime_tensor rhs;
    runtime_tensor con;
};

INSTANTIATE_TEST_SUITE_P(Where, WhereTest,
                         testing::Combine(testing::Range(0, 1000)));

TEST_P(WhereTest, Where) {
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

    // expected
    size_t size = 0;
    auto output_ort =
        ortki_Where(runtime_tensor_2_ort_tensor(con), l_ort, r_ort);
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::where(true, con.impl(), lhs.impl(), rhs.impl())
            .expect("where failed");
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
    FOR_LOOP(lhs_type, j)
    FOR_LOOP(rhs_type, i)
    FOR_LOOP(i_shape, k)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(rhs_type, i)
    SPLIT_ELEMENT(i_shape, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}