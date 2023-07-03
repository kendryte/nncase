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

class GatherTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape] = GetParam();

        //        size_t size = 0;
        int32_t input_array[] = {0, 1, 2, 3};
        input = hrt::create(dt_int32, shape,
                            {reinterpret_cast<gsl::byte *>(input_array),
                             sizeof(input_array)},
                            true, host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");

        int64_t indices_array[] = {0, 0, 1, 1};
        indices = hrt::create(dt_int64, shape,
                              {reinterpret_cast<gsl::byte *>(indices_array),
                               sizeof(indices_array)},
                              true, host_runtime_tensor::pool_cpu_only)
                      .expect("create tensor failed");

        int64_t batchDims_array[1] = {0};
        batchDims = hrt::create(dt_int64, dims_t{1},
                                {reinterpret_cast<gsl::byte *>(batchDims_array),
                                 sizeof(batchDims_array)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    }

    void TearDown() override {}

    void init_tensor(runtime::runtime_tensor &tensor) override {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            int8_t fixed_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    size_t flat_index = 0;
                    for (size_t i = 0; i < index.size(); i++) {
                        flat_index += index[i] * tensor.strides()[i];
                    }
                    get<int8_t>(tensor, index) = fixed_values[flat_index % 10];
                    return ok();
                });
            break;
        }
        case dt_int16: {
            int16_t fixed_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    size_t flat_index = 0;
                    for (size_t i = 0; i < index.size(); i++) {
                        flat_index += index[i] * tensor.strides()[i];
                    }
                    get<int16_t>(tensor, index) = fixed_values[flat_index % 10];
                    return ok();
                });
            break;
        }
        case dt_int32: {
            int32_t fixed_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    size_t flat_index = 0;
                    for (size_t i = 0; i < index.size(); i++) {
                        flat_index += index[i] * tensor.strides()[i];
                    }
                    get<int32_t>(tensor, index) = fixed_values[flat_index % 10];
                    return ok();
                });
            break;
        }
        case dt_float32: {
            float fixed_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    size_t flat_index = 0;
                    for (size_t i = 0; i < index.size(); i++) {
                        flat_index += index[i] * tensor.strides()[i];
                    }
                    get<float>(tensor, index) = fixed_values[flat_index % 10];
                    return ok();
                });
            break;
        }
        case dt_float16: {
            half fixed_values[] = {(half)1, (half)2, (half)3, (half)4, (half)5, (half)6, (half)7, (half)8, (half)9, (half)10};
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    size_t flat_index = 0;
                    for (size_t i = 0; i < index.size(); i++) {
                        flat_index += index[i] * tensor.strides()[i];
                    }
                    get<float>(tensor, index) = fixed_values[flat_index % 10];
                    return ok();
                });
            break;
        }
        default: {
            break;
        }
        }
    }

  protected:
    runtime_tensor input;
    runtime_tensor indices;
    runtime_tensor batchDims;
};

INSTANTIATE_TEST_SUITE_P(Gather, GatherTest,
                         testing::Combine(testing::Values(dt_int32, dt_int64),
                                          testing::Values(dims_t{2, 2})));

TEST_P(GatherTest, gather) {
    auto input_ort = runtime_tensor_2_ort_tensor(input);
    auto indices_ort = runtime_tensor_2_ort_tensor(indices);

    // expected
    auto output_ort = ortki_Gather(input_ort, indices_ort, 0);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::gather(input.impl(), batchDims.impl(), indices.impl())
            .expect("gather failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, actual));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}