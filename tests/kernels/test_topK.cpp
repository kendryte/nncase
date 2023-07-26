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

class TopKTest
    : public KernelTest,
      public ::testing::TestWithParam<
          std::tuple<nncase::typecode_t, dims_t, int64_t, int64_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, value1, value2, value3] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        axis_value = value1;
        int64_t axis_array[] = {value1};
        axis = hrt::create(dt_int64, {1},
                           {reinterpret_cast<gsl::byte *>(axis_array),
                            sizeof(axis_array)},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");

        largest_value = value2;
        int64_t largest_array[] = {value2};
        largest = hrt::create(dt_int64, {1},
                              {reinterpret_cast<gsl::byte *>(largest_array),
                               sizeof(largest_array)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        sorted_value = value3;
        int64_t sorted_array[] = {value3};
        sorted = hrt::create(dt_int64, {1},
                             {reinterpret_cast<gsl::byte *>(sorted_array),
                              sizeof(sorted_array)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    int64_t axis_value;
    runtime_tensor axis;
    int64_t largest_value;
    runtime_tensor largest;
    int64_t sorted_value;
    runtime_tensor sorted;
};

INSTANTIATE_TEST_SUITE_P(
    TopK, TopKTest,
    testing::Combine(testing::Values(dt_float32),
                     testing::Values(dims_t{1, 2, 4, 8}, dims_t{1, 3, 16, 16},
                                     dims_t{3, 3, 6}, dims_t{16, 16}, dims_t{1},
                                     dims_t{1, 3}),
                     testing::Values(0, -1), testing::Values(0, 1),
                     testing::Values(0, 1)));

TEST_P(TopKTest, TopK) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);

    // expected
    size_t size = 0;
    int64_t k_array[] = {1};
    auto k =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(k_array), sizeof(k_array)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto output_ort1 = tensor_seq_get_value(
        ortki_TopK(l_ort, runtime_tensor_2_ort_tensor(k), axis_value,
                   largest_value, sorted_value),
        0);
    void *ptr_ort1 = tensor_buffer(output_ort1, &size);
    dims_t shape1(tensor_rank(output_ort1));
    tensor_shape(output_ort1, reinterpret_cast<int64_t *>(shape1.data()));
    auto expected1 =
        hrt::create(input.datatype(), shape1,
                    {reinterpret_cast<gsl::byte *>(ptr_ort1), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    size = 0;
    auto output_ort2 = tensor_seq_get_value(
        ortki_TopK(l_ort, runtime_tensor_2_ort_tensor(k), axis_value,
                   largest_value, sorted_value),
        1);
    void *ptr_ort2 = tensor_buffer(output_ort2, &size);
    dims_t shape2(tensor_rank(output_ort2));
    tensor_shape(output_ort2, reinterpret_cast<int64_t *>(shape2.data()));
    auto expected2 =
        hrt::create(dt_int64, shape2,
                    {reinterpret_cast<gsl::byte *>(ptr_ort2), size}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    runtime_tensor expected[] = {expected1, expected2};

    // actual
    auto output = kernels::stackvm::top_k(input.impl(), k.impl(), axis.impl(),
                                          largest.impl(), sorted.impl())
                      .expect("topk failed");
    [[maybe_unused]] auto actual(output.as<tuple>().expect("as tensor failed"));

    typecode_t dtypes[] = {dt_float32, dt_int64};
    [[maybe_unused]] auto result = check_tuple_output(expected, dtypes, output);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}