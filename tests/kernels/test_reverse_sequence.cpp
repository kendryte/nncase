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

class ReverseSequenceTest
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

INSTANTIATE_TEST_SUITE_P(ReverseSequence, ReverseSequenceTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{4, 4})));

TEST_P(ReverseSequenceTest, ReverseSequence) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    size_t size = 0;
    int64_t seqLens_array[] = {1, 2, 3, 4};
    auto seqLens = hrt::create(dt_int64, {4},
                               {reinterpret_cast<gsl::byte *>(seqLens_array),
                                sizeof(seqLens_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");
    auto output_ort = ortki_ReverseSequence(
        l_ort, runtime_tensor_2_ort_tensor(seqLens), 1, 0);
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    int64_t batch_axis_array[] = {1};
    auto batch_axis =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(batch_axis_array),
                     sizeof(batch_axis_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    int64_t time_axis_array[] = {0};
    auto time_axis =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(time_axis_array),
                     sizeof(time_axis_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output =
        kernels::stackvm::reverse_sequence(input.impl(), seqLens.impl(),
                                           batch_axis.impl(), time_axis.impl())
            .expect("reverse_sequence failed");
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
    FOR_LOOP(rhs_type, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(lhs_type, j)
    SPLIT_ELEMENT(rhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}