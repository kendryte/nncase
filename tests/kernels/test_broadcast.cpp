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

class BroadCastTest : public KernelTest,
                      public ::testing::TestWithParam<
                          std::tuple<nncase::typecode_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, l_shape, r_shape] = GetParam();

        input =
            hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        one = hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor_one(one);

        size_t shape_size = r_shape.size();
        int64_t *shape_array = (int64_t *)malloc(shape_size * sizeof(int64_t));
        std::copy(r_shape.begin(), r_shape.end(), shape_array);
        new_shape = hrt::create(dt_int64, {shape_size},
                                {reinterpret_cast<gsl::byte *>(shape_array),
                                 shape_size * sizeof(int64_t)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
    }

    void TearDown() override {}

    void init_tensor_one(runtime::runtime_tensor &tensor) {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int8_t>(tensor, index) = static_cast<int8_t>(1);
                    return ok();
                });
            break;
        }
        case dt_int16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int16_t>(tensor, index) = static_cast<int16_t>(1);
                    return ok();
                });
            break;
        }
        case dt_int32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int32_t>(tensor, index) = 1;
                    return ok();
                });
            break;
        }
        case dt_int64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int64_t>(tensor, index) = static_cast<int64_t>(1);
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint8_t>(tensor, index) = static_cast<uint8_t>(1);
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint16_t>(tensor, index) = static_cast<uint16_t>(1);
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint32_t>(tensor, index) = static_cast<uint32_t>(1);
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint64_t>(tensor, index) = static_cast<uint64_t>(1);
                    return ok();
                });
            break;
        }
        case dt_float16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<half>(tensor, index) = static_cast<half>(1);
                    return ok();
                });
            break;
        }
        case dt_float32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(1);
                    return ok();
                });
            break;
        }
        case dt_float64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<double>(tensor, index) = static_cast<double>(1);
                    return ok();
                });
            break;
        }
        case dt_bfloat16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<bfloat16>(tensor, index) = static_cast<bfloat16>(1);
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

  protected:
    runtime_tensor input;
    runtime_tensor one;
    runtime_tensor new_shape;
};

INSTANTIATE_TEST_SUITE_P(
    BroadCast, BroadCastTest,
    testing::Combine(testing::Values(dt_float32, dt_float64, dt_int32, dt_int64,
                                     dt_float16),
                     testing::Values(dims_t{3}, dims_t{1, 3}, dims_t{3, 3},
                                     dims_t{1}, dims_t{1, 3, 1}),
                     testing::Values(dims_t{1, 3, 3}, dims_t{1, 3, 3, 3},
                                     dims_t{1, 3, 16, 16})));

TEST_P(BroadCastTest, BroadCast) {

    // expected
    auto output_ort = ortki_Mul(runtime_tensor_2_ort_tensor(input),
                                runtime_tensor_2_ort_tensor(one));
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::broadcast(input.impl(), new_shape.impl())
                      .expect("broadcast failed");
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