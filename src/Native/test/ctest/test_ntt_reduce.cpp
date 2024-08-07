/* Copyright 2019-2024 Canaan Inc.
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
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;

TEST(ReduceSumTestFloat, ReduceM_NoPack) {
    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1024, 1024>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 1024>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::reduce<ntt::ops::add>(*ntt_input, *ntt_output1, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t buf[] = {0};
    int64_t shape[1] = {1};
    auto axes =
        make_tensor(reinterpret_cast<void *>(buf), DataType_INT64, shape, 1);
    auto ort_output = ortki_ReduceSum(ort_input, axes, 1, 1);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ReduceMaxTestFloat, ReduceM_NoPack) {
    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1024, 1024>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 1024>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::reduce<ntt::ops::max>(*ntt_input, *ntt_output1, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t buf[] = {0};
    int64_t shape[1] = {1};
    auto axes =
        make_tensor(reinterpret_cast<void *>(buf), DataType_INT64, shape, 1);
    auto ort_output = ortki_ReduceMax(ort_input, axes, 1, 1);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ReduceMinTestFloat, ReduceM_NoPack) {
    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1024, 1024>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 1024>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::reduce<ntt::ops::min>(*ntt_input, *ntt_output1, ntt::fixed_shape<0>{},
                               ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t buf[] = {0};
    int64_t shape[1] = {1};
    auto axes =
        make_tensor(reinterpret_cast<void *>(buf), DataType_INT64, shape, 1);
    auto ort_output = ortki_ReduceMin(ort_input, axes, 1, 1);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

TEST(ReduceMeanTestFloat, ReduceM_NoPack) {
    // init
    using tensor_type1 = ntt::tensor<float, ntt::fixed_shape<1024, 1024>>;
    std::unique_ptr<tensor_type1> ntt_input(new tensor_type1);
    NttTest::init_tensor(*ntt_input, -10.f, 10.f);

    // ntt
    using tensor_type2 = ntt::tensor<float, ntt::fixed_shape<1, 1024>>;
    std::unique_ptr<tensor_type2> ntt_output1(new tensor_type2);
    ntt::reduce<ntt::ops::mean>(*ntt_input, *ntt_output1, ntt::fixed_shape<0>{},
                                ntt::fixed_shape<>{}, ntt::fixed_shape<>{});

    // ort
    auto ort_input = NttTest::ntt2ort(*ntt_input);
    int64_t buf[] = {0};
    int64_t shape[1] = {1};
    auto axes =
        make_tensor(reinterpret_cast<void *>(buf), DataType_INT64, shape, 1);
    auto ort_output = ortki_ReduceMean(ort_input, axes, 1, 1);

    // compare
    std::unique_ptr<tensor_type2> ntt_output2(new tensor_type2);
    NttTest::ort2ntt(ort_output, *ntt_output2);
    EXPECT_TRUE(NttTest::compare_tensor(*ntt_output1, *ntt_output2));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
