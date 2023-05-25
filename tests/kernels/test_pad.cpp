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
#include <c_api.h>
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <operators.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class PadTest : public KernelTest,
                public ::testing::TestWithParam<
                    std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
};

INSTANTIATE_TEST_SUITE_P(Pad, PadTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 1, 2, 3})));

TEST_P(PadTest, Pad) {

    // expected
    size_t size = 0;
    int64_t pad_ptr[] = { 0, 0, 1, 2, 2, 4, 5, 6 };
    auto pad = hrt::create(dt_int64, {4, 2},
                           {reinterpret_cast<gsl::byte *>(pad_ptr), size},
                           true, host_runtime_tensor::pool_cpu_only)
                   .expect("create tensor failed");
    float value_ptr[] = {1.0f};
    auto value = hrt::create(dt_float32, {1},
                             {reinterpret_cast<gsl::byte *>(value_ptr), size},
                             true, host_runtime_tensor::pool_cpu_only)
                     .expect("create tensor failed");
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    auto pad_ort = runtime_tensor_2_ort_tensor(pad);
    auto value_ort = runtime_tensor_2_ort_tensor(value);
    auto output_ort = ortki_Pad(l_ort, pad_ort, value_ort, "constant");
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::pad(runtime::stackvm::pad_mode_t::constant, lhs.impl(), pad.impl(), value.impl())
            .expect("pad failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}