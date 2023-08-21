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

#define TEST_CASE_NAME "test_layer_norm"

using namespace nncase;
using namespace nncase::runtime;
using namespace ortki;

class LayerNormTest : public KernelTest,
                      public ::testing::TestWithParam<std::tuple<int>> {
  public:
    void SetUp() override {
        READY_SUBCASE()

        auto l_shape = GetShapeArray("lhs_shape");
        auto axis = GetNumber("axis");
        auto typecode = GetDataType("lhs_type");

        axis_value = axis > (int64_t)l_shape.size() - 1
                         ? (int64_t)l_shape.size() - 1
                         : axis;

        axis_value =
            (axis_value < 0 && (-axis_value) > (int64_t)l_shape.size() - 1)
                ? (1 - (int64_t)l_shape.size())
                : axis_value;

        input =
            hrt::create(typecode, l_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(input);

        int64_t axis1 =
            axis_value < 0 ? axis_value + (int64_t)l_shape.size() : axis_value;

        size_t l_shape_sum = 1;
        for (size_t i = axis1; i < l_shape.size(); i++) {
            l_shape_sum = l_shape_sum * l_shape[i];
        }

        dims_t scale_shape = {l_shape_sum};
        scale = hrt::create(typecode, scale_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(scale);

        dims_t b_shape = {l_shape_sum};
        b = hrt::create(typecode, b_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(b);
    }

    void TearDown() override { CLEAR_SUBCASE() }

  protected:
    runtime_tensor input;
    runtime_tensor scale;
    runtime_tensor b;
    int64_t axis_value;
};

INSTANTIATE_TEST_SUITE_P(layer_norm, LayerNormTest,
                         testing::Combine(testing::Range(0, MAX_CASE_NUM)));

TEST_P(LayerNormTest, layer_norm) {
    auto l_ort = runtime_tensor_2_ort_tensor(input);
    auto scale_ort = runtime_tensor_2_ort_tensor(scale);
    auto b_ort = runtime_tensor_2_ort_tensor(b);
    auto eps = 1e-05f;

    //     expected
    auto output_ort =
        ortki_LayerNormalization(l_ort, scale_ort, b_ort, axis_value, eps, 1L);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(tensor_seq_get_value(output_ort, 0), &size);
    dims_t shape(tensor_rank(tensor_seq_get_value(output_ort, 0)));
    tensor_shape(tensor_seq_get_value(output_ort, 0),
                 reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(input.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    auto output =
        kernels::stackvm::layer_norm((int32_t)axis_value, eps, input.impl(),
                                     scale.impl(), b.impl())
            .expect("layer_norm failed");
    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    bool result = is_same_tensor(expected, actual) ||
                  cosine_similarity_tensor(expected, actual);

    if (!result) {
        print_runtime_tensor(actual);
        print_runtime_tensor(expected);
    }

    // compare
    EXPECT_TRUE(result);
}

int main(int argc, char *argv[]) {
    READY_TEST_CASE_GENERATE()
    FOR_LOOP(lhs_shape, i)
    FOR_LOOP(axis, j)
    FOR_LOOP(lhs_type, k)
    SPLIT_ELEMENT(lhs_shape, i)
    SPLIT_ELEMENT(axis, j)
    SPLIT_ELEMENT(lhs_type, k)
    WRITE_SUB_CASE()
    FOR_LOOP_END()
    FOR_LOOP_END()
    FOR_LOOP_END()

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}