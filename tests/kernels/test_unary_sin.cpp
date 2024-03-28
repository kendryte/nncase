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

#define TEST_CASE_NAME "test_unary_other_type1"

class UnaryTest : public KernelTest,
                  public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto typecode = GetDataType("other_type");
        auto l_shape = GetShapeArray("i_shape");

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);
    }

    void TearDown() override { CLEAR_SUBCASE() }

    void init_tensor(runtime_tensor &tensor) override {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<int8_t>(tensor, index) = static_cast<int8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<int16_t>(tensor, index) =
                        static_cast<int16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_int32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<int32_t>(tensor, index) = dis(gen);
                    return ok();
                });
            break;
        }
        case dt_int64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<int64_t>(tensor, index) =
                        static_cast<int64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<uint8_t>(tensor, index) =
                        static_cast<uint8_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<uint16_t>(tensor, index) =
                        static_cast<uint16_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<uint32_t>(tensor, index) =
                        static_cast<uint32_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(-100000, 100000);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<uint64_t>(tensor, index) =
                        static_cast<uint64_t>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float16: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-10000.0f, 10000.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<half>(tensor, index) = static_cast<half>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float32: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-100000.0f, 100000.0f);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(dis(gen));
                    return ok();
                });
            break;
        }
        case dt_float64: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-100000.0, 100000.0);
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](std::span<const size_t> index) -> result<void> {
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

INSTANTIATE_TEST_SUITE_P(Unary, UnaryTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(UnaryTest, sin) {
    OrtKITensor *orts[1];
    orts[0] = runtime_tensor_2_ort_tensor(input);

    // expected
    auto output_ort = ortki_Sin(orts[0]);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<std::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output = kernels::stackvm::unary(
                      nncase::runtime::stackvm::unary_op_t::sin, input.impl())
                      .expect("unary failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        std::cout << "input ";
        print_runtime_tensor(input);
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
    FOR_LOOP(other_type, i)
    FOR_LOOP(i_shape, j)
    SPLIT_ELEMENT(other_type, i)
    SPLIT_ELEMENT(i_shape, j)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}