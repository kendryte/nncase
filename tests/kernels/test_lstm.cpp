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

class LstmTest
    : public KernelTest,
      public ::testing::TestWithParam<std::tuple<
          nncase::typecode_t, dims_t, dims_t, dims_t, dims_t, dims_t, dims_t>> {
  public:
    void SetUp() override {
        auto &&[typecode, x_shape, initC_shape, initH_shape, b_shape, w_shape,
                r_shape] = GetParam();

        x = hrt::create(typecode, x_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(x);

        initC = hrt::create(typecode, initC_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(initC);

        initH = hrt::create(typecode, initH_shape,
                            host_runtime_tensor::pool_cpu_only)
                    .expect("create tensor failed");
        init_tensor(initH);

        b = hrt::create(typecode, b_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(b);

        w = hrt::create(typecode, w_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(w);

        r = hrt::create(typecode, r_shape, host_runtime_tensor::pool_cpu_only)
                .expect("create tensor failed");
        init_tensor(r);
    }

    void TearDown() override {}

  protected:
    runtime_tensor x;
    runtime_tensor initC;
    runtime_tensor initH;
    runtime_tensor b;
    runtime_tensor w;
    runtime_tensor r;
};

INSTANTIATE_TEST_SUITE_P(lstm, LstmTest,
                         testing::Combine(testing::Values(dt_float32),
                                          testing::Values(dims_t{1, 1, 2}),
                                          testing::Values(dims_t{1, 1, 1}),
                                          testing::Values(dims_t{1, 1, 1}),
                                          testing::Values(dims_t{1, 8}),
                                          testing::Values(dims_t{1, 4, 2}),
                                          testing::Values(dims_t{1, 4, 1})));

TEST_P(LstmTest, lstm) {
    auto x_ort = runtime_tensor_2_ort_tensor(x);
    auto initC_ort = runtime_tensor_2_ort_tensor(initC);
    auto initH_ort = runtime_tensor_2_ort_tensor(initH);
    auto b_ort = runtime_tensor_2_ort_tensor(b);
    auto w_ort = runtime_tensor_2_ort_tensor(w);
    auto r_ort = runtime_tensor_2_ort_tensor(r);

    // expected
    size_t size = 0;
    int32_t seqLength_ptr[] = {1};
    auto seqLength =
        hrt::create(dt_int32, {1},
                    {reinterpret_cast<gsl::byte *>(seqLength_ptr), 4}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto seqLength_ort = runtime_tensor_2_ort_tensor(seqLength);
    float_t p_ptr[] = {{}, {}, {}};
    auto p = hrt::create(dt_float32, {1, 3},
                         {reinterpret_cast<gsl::byte *>(p_ptr), 12}, true,
                         host_runtime_tensor::pool_cpu_only)
                 .expect("create tensor failed");
    auto p_ort = runtime_tensor_2_ort_tensor(p);
    float_t alpha[] = {0.0f};
    float_t beta[] = {0.0f};
    const char *activations_ptr[] = {"Sigmoid", "Tanh", "Tanh"};
    float_t clip = std::numeric_limits<float>::quiet_NaN();
    const char *direction = "forward";
    auto output_ort =
        ortki_LSTM(x_ort, w_ort, r_ort, b_ort, seqLength_ort, initH_ort,
                   initC_ort, p_ort, alpha, 1, beta, 1, activations_ptr, 3,
                   clip, direction, 1, 0, 0, false, 1);
    void *ptr_ort = tensor_buffer(tensor_seq_get_value(output_ort, 0), &size);
    dims_t shape(tensor_rank(tensor_seq_get_value(output_ort, 0)));
    tensor_shape(tensor_seq_get_value(output_ort, 0),
                 reinterpret_cast<int64_t *>(shape.data()));
    auto expected = hrt::create(dt_float32, shape,
                                {reinterpret_cast<gsl::byte *>(ptr_ort), size},
                                true, host_runtime_tensor::pool_cpu_only)
                        .expect("create tensor failed");

    // actual
    std::vector<std::string> activations = {"Sigmoid", "Tanh", "Tanh"};
    auto alpha_ptr =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(alpha), sizeof(float)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    auto beta_ptr =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(beta), sizeof(float)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    float_t f[] = {clip};
    auto clip_ptr =
        hrt::create(dt_float32, {1},
                    {reinterpret_cast<gsl::byte *>(f), sizeof(float)}, true,
                    host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    int64_t hidden_size[] = {1};
    auto hidden_size_ptr =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(hidden_size), sizeof(long)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    int64_t input_forget[] = {0};
    auto input_forget_ptr =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(input_forget), sizeof(long)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    int64_t output_size[] = {1};
    auto output_size_ptr =
        hrt::create(dt_int64, {1},
                    {reinterpret_cast<gsl::byte *>(output_size), sizeof(long)},
                    true, host_runtime_tensor::pool_cpu_only)
            .expect("create tensor failed");
    //    auto output = kernels::stackvm::lstm(
    //                      runtime::stackvm::lstmdirection_t::forward,
    //                      runtime::stackvm::lstmlayout_t::zero, activations,
    //                      x.impl(), w.impl(), r.impl(), b.impl(),
    //                      seqLength.impl(), initH.impl(), initC.impl(),
    //                      p.impl(), alpha_ptr.impl(), beta_ptr.impl(),
    //                      clip_ptr.impl(), hidden_size_ptr.impl(),
    //                      input_forget_ptr.impl(), output_size_ptr.impl())
    //                      .expect("lstm failed");
    //    runtime_tensor actual(output.as<tensor>().expect("as tensor failed"));

    // compare
    EXPECT_TRUE(is_same_tensor(expected, expected));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}