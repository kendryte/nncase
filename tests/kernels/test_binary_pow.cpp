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

class BinaryTest : public KernelTest,
                   public ::testing::TestWithParam<
                       std::tuple<nncase::typecode_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, r_shape] = GetParam();

        lhs = hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        rhs = hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");

        if (typecode == dt_float32) {
            init_tensor_pow_f32(lhs);
            init_tensor_pow_f32(rhs);
        } else {
            init_tensor(lhs);
            init_tensor(rhs);
        }
    }

    void TearDown() override {}

    virtual void init_tensor_pow_f32(runtime::runtime_tensor &tensor) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-6.0f, 6.0f);
        NNCASE_UNUSED auto res = kernels::stackvm::apply(
            tensor.shape(), [&](const dims_t &index) -> result<void> {
                get<float>(tensor, index) = static_cast<int32_t>(dis(gen));
                return ok();
            });
    }

  protected:
    runtime_tensor lhs;
    runtime_tensor rhs;
};

INSTANTIATE_TEST_SUITE_P(
    Binary, BinaryTest,
    testing::Combine(testing::Values(dt_int32, dt_int64, dt_float32),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{3, 16, 16},
                                     dims_t{3, 16, 1}, dims_t{16, 16},
                                     dims_t{16, 1}, dims_t{1, 16, 1},
                                     dims_t{16}, dims_t{1}, dims_t{}),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{3, 16, 1},
                                     dims_t{3, 16, 1}, dims_t{16, 16},
                                     dims_t{1, 16, 1}, dims_t{16, 1},
                                     dims_t{16}, dims_t{1}, dims_t{})));

TEST_P(BinaryTest, pow) {
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);

    // expected
    auto output_ort = ortki_Pow(l_ort, r_ort);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::binary(nncase::runtime::stackvm::binary_op_t::pow,
                                 lhs.impl(), rhs.impl())
            .expect("binary failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual) ||
                cosine_similarity_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}