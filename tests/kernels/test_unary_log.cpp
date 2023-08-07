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

class UnaryTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<nncase::typecode_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, i_shape] = GetParam();

        input =
            hrt::create(typecode, i_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override {}

    void init_tensor(runtime_tensor &tensor) override {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int8_t>(tensor, index) = static_cast<int8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int16_t>(tensor, index) =
                        static_cast<int16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int32_t>(tensor, index) = dis(gen);
                    return ok();
                });
            break;
        }
        case dt_int64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 6);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int64_t>(tensor, index) =
                        static_cast<int64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint8_t>(tensor, index) =
                        static_cast<uint8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint16_t>(tensor, index) =
                        static_cast<uint16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint32_t>(tensor, index) =
                        static_cast<uint32_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(1, 127);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint64_t>(tensor, index) =
                        static_cast<uint64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(1.0f, 2.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(1.0, 2.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<double>(tensor, index) = static_cast<double>(dis(gen));
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
};

INSTANTIATE_TEST_SUITE_P(
    Unary, UnaryTest,
    testing::Combine(testing::Values(dt_float32,
                                     /* dt_int32, dt_int64,*/ dt_float64,
                                     dt_float16), // onnx no support
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{3, 16, 16},
                                     dims_t{3, 16, 1}, dims_t{16, 16},
                                     dims_t{16, 1}, dims_t{1, 16, 1},
                                     dims_t{16}, dims_t{1}, dims_t{})));

TEST_P(UnaryTest, log) {
    OrtKITensor *orts[1];
    orts[0] = runtime_tensor_2_ort_tensor(input);

    // expected
    auto output_ort = ortki_Log(orts[0]);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::unary(
                      nncase::runtime::stackvm::unary_op_t::log, input.impl())
                      .expect("unary failed");
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