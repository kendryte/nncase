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

class ConcatTest : public KernelTest,
                   public ::testing::TestWithParam<
                       std::tuple<nncase::typecode_t, dims_t, int64_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, shape, value] = GetParam();

        lhs = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(lhs);

        rhs = hrt::create(typecode, shape, host_runtime_tensor::pool_cpu_only)
                  .expect("create tensor failed");
        init_tensor(rhs);

        axis_value = value > 0 ? value >= (int64_t)shape.size() ? 0 : value
                     : -value > (int64_t)shape.size() ? 0
                                                      : value;
    }

    void TearDown() override {}

  protected:
    runtime_tensor lhs;
    runtime_tensor rhs;
    int64_t axis_value;
};

INSTANTIATE_TEST_SUITE_P(
    Concat, ConcatTest,
    testing::Combine(testing::Values(dt_float32, dt_int64, dt_int32, dt_float64,
                                     dt_float16, dt_int16, dt_uint32,
                                     dt_boolean),
                     testing::Values(dims_t{1, 3, 16, 16}, dims_t{1, 3},
                                     dims_t{1, 3, 16}, dims_t{1}),
                     testing::Values(-1, 0, 1, 2, 3, -2, -3, -4)));

TEST_P(ConcatTest, Concat) {
    auto l_ort = runtime_tensor_2_ort_tensor(lhs);
    auto r_ort = runtime_tensor_2_ort_tensor(rhs);
    OrtKITensor *ls_ort[2] = {l_ort, r_ort};

    // expected
    auto output_ort = ortki_Concat(ls_ort, 2, axis_value);
    size_t size = 0;
    void *ptr_ort = tensor_buffer(output_ort, &size);
    dims_t shape(tensor_rank(output_ort));
    tensor_shape(output_ort, reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(lhs.datatype(), shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    value_t field1 = lhs.impl();
    value_t field2 = rhs.impl();
    std::vector<value_t> fields;
    fields.push_back(field1);
    fields.push_back(field2);
    auto output_tuple = tuple(std::in_place, std::move(fields));

    int64_t axis_ptr[] = {axis_value};
    auto axis =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(axis_ptr), sizeof(axis_ptr)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");

    auto output = kernels::stackvm::concat(output_tuple, axis.impl())
                      .expect("concat failed");

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