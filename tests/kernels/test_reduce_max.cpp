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

#define TEST_CASE_NAME "test_reduce"

class ReduceMaxTest : public KernelTest,
                      public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {

        READY_SUBCASE()

        auto typecode1 = GetDataType("lhs_type");
        auto typecode2 = GetDataType("rhs_type");
        auto l_shape = GetShapeArray("lhs_shape");
        auto r_shape = GetShapeArray("rhs_shape");
        auto value = GetNumber("bool_value");
        auto axis_value = GetAxesArray("axis_value");

        a = hrt::create(typecode1, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(a);

        keepDims_value = value;
        int64_t keepDims_array[] = {keepDims_value};
        keepDims = hrt::create(typecode2, r_shape,
                               {reinterpret_cast<gsl::byte *>(keepDims_array),
                                sizeof(keepDims_array)},
                               true, host_runtime_tensor::pool_cpu_only)
                       .expect("create tensor failed");

        init_value =
            hrt::create(typecode1, r_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_value_tensor(init_value);

        axis_value_array = axis_value;
    }

    void TearDown() override { CLEAR_SUBCASE() }

    virtual void init_value_tensor(runtime::runtime_tensor &tensor) {
        auto dtype = tensor.datatype();
        switch (dtype) {
        case dt_int8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int8_t>(tensor, index) = static_cast<int8_t>(-6);
                    return ok();
                });
            break;
        }
        case dt_int16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int16_t>(tensor, index) = static_cast<int16_t>(-6);
                    return ok();
                });
            break;
        }
        case dt_int32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int32_t>(tensor, index) = -6;
                    return ok();
                });
            break;
        }
        case dt_int64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<int64_t>(tensor, index) = static_cast<int64_t>(-6);
                    return ok();
                });
            break;
        }
        case dt_uint8: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint8_t>(tensor, index) = static_cast<uint8_t>(0);
                    return ok();
                });
            break;
        }
        case dt_uint16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint16_t>(tensor, index) = static_cast<uint16_t>(0);
                    return ok();
                });
            break;
        }
        case dt_uint32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint32_t>(tensor, index) = static_cast<uint32_t>(0);
                    return ok();
                });
            break;
        }
        case dt_uint64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<uint64_t>(tensor, index) = static_cast<uint64_t>(0);
                    return ok();
                });
            break;
        }
        case dt_float16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<half>(tensor, index) = static_cast<half>(-1);
                    return ok();
                });
            break;
        }
        case dt_float32: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<float>(tensor, index) = static_cast<float>(-1);
                    return ok();
                });
            break;
        }
        case dt_float64: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<double>(tensor, index) = static_cast<double>(-1);
                    return ok();
                });
            break;
        }
        case dt_bfloat16: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<bfloat16>(tensor, index) = static_cast<bfloat16>(-1);
                    return ok();
                });
            break;
        }
        case dt_boolean: {
            NNCASE_UNUSED auto res = kernels::stackvm::apply(
                tensor.shape(),
                [&](gsl::span<const size_t> index) -> result<void> {
                    get<bool>(tensor, index) = false;
                    return ok();
                });
            break;
        }
        default: {
        }
        }
    }

  protected:
    runtime_tensor a;
    axes_t axis_value_array;
    int64_t keepDims_value;
    runtime_tensor keepDims;
    runtime_tensor init_value;
};

INSTANTIATE_TEST_SUITE_P(ReduceMax, ReduceMaxTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(ReduceMaxTest, ReduceMax) {

    size_t axis_size = axis_value_array.size();
    if (axis_size <= a.shape().size()) {
        int64_t *axis_array = (int64_t *)malloc(axis_size * sizeof(int64_t));
        std::copy(axis_value_array.begin(), axis_value_array.end(), axis_array);
        auto axis = hrt::create(dt_int64, {axis_size},
                                {reinterpret_cast<gsl::byte *>(axis_array),
                                 axis_size * sizeof(int64_t)},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");
        auto axis_ort = runtime_tensor_2_ort_tensor(axis);
        auto output_ort = ortki_ReduceMax(runtime_tensor_2_ort_tensor(a),
                                          axis_ort, keepDims_value, 0);

        // expected
        size_t size = 0;
        void *ptr_ort = tensor_buffer(output_ort, &size);
        dims_t shape(tensor_rank(output_ort));
        tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
        auto expected =
            hrt::create(a.datatype(), shape,
                        {reinterpret_cast<gsl::byte *>(ptr_ort), size}, true,
                        host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");

        // actual
        auto output = kernels::stackvm::reduce(
                          runtime::stackvm::reduce_op_t::max, a.impl(),
                          axis.impl(), init_value.impl(), keepDims.impl())
                          .expect("reduce_max failed");
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
        free(axis_array);
    }
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_type, i)
    FOR_LOOP(rhs_type, j)
    FOR_LOOP(lhs_shape, k)
    FOR_LOOP(rhs_shape, l)
    FOR_LOOP(bool_value, m)
    FOR_LOOP(axis_value, n)
    SPLIT_ELEMENT(lhs_type, i)
    SPLIT_ELEMENT(rhs_type, j)
    SPLIT_ELEMENT(lhs_shape, k)
    SPLIT_ELEMENT(rhs_shape, l)
    SPLIT_ELEMENT(bool_value, m)
    SPLIT_ELEMENT(axis_value, n)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}