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

class TriluTest
    : public KernelTest,
      public ::testing::TestWithParam<
          std::tuple<nncase::typecode_t, dims_t, int64_t, int32_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, k_value, upper_value] = GetParam();

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        int64_t k_ptr[] = {k_value};
        k = hrt::create(nncase::dt_int64, {1},
                        {reinterpret_cast<gsl::byte *>(k_ptr), sizeof(k_ptr)},
                        true, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        upper = upper_value;
    }

    void TearDown() override {}

  protected:
    runtime_tensor input;
    runtime_tensor k;
    int32_t upper;
};

INSTANTIATE_TEST_SUITE_P(trilu, TriluTest,
                         testing::Combine(testing::Values(dt_uint8),
                                          testing::Values(dims_t{4, 5}),
                                          testing::Values(0),
                                          testing::Values(1)));

TEST_P(TriluTest, trilu) {
    //    auto l_ort = runtime_tensor_2_ort_tensor(input);
    //    auto k_ort = runtime_tensor_2_ort_tensor(k);
    //
    //    // expected
    //    auto output_ort = ortki_Trilu(l_ort, k_ort, upper);
    //    size_t size = 0;
    //    void *ptr_ort = tensor_buffer(output_ort, &size);
    //    dims_t shape(tensor_rank(output_ort));
    //    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    //    auto expected = hrt::create(input.datatype(), shape,
    //                                {reinterpret_cast<gsl::byte *>(ptr_ort),
    //                                size}, true,
    //                                host_runtime_tensor::pool_cpu_only)
    //                        .expect("create tensor failed");

    // actual
    int32_t upper_ptr[] = {upper};
    auto upper = hrt::create(nncase::dt_int32, {1},
                             {reinterpret_cast<gsl::byte *>(upper_ptr),
                              sizeof(upper_ptr)},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");

    auto output = kernels::stackvm::trilu(input.impl(), k.impl(), upper.impl())
                      .expect("trilu failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    //    bool result = is_same_tensor(expected, actual) ||
    //                  cosine_similarity_tensor(expected, actual);
    //
    //    if (!result) {
    //        std::cout << "actual ";
    //        print_runtime_tensor(actual);
    //        std::cout << "expected ";
    //        print_runtime_tensor(expected);
    //    }
    //
    //    // compare
    //    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}