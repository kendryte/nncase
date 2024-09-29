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
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <nncase/api.h>
#include <nncase/compiler.h>
#include <nncase/io_utils.h>
#include <nncase/ntt/ntt.h>
#include <string_view>

using namespace nncase;
using namespace nncase::clr;
using namespace nncase::runtime;
using namespace std::string_view_literals;

bool are_floats_equal(float a, float b, float epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

#define TRY(x)                                                                 \
    if (x)                                                                     \
        throw 1;

int main() {
#if 0
    nncase_clr_initialize(
        R"(E:\Work\Repos\nncase-v2\nncase\src\Nncase.Compiler\bin\Debug\net6.0\Nncase.Compiler.dll)");
    auto target_name = "cpu"sv;
    auto nncapi = nncase_clr_api();
    clr_object_ptr target, compile_session, compiler, compile_options;
    compile_options = nncapi->compile_options_create();
    target = nncapi->target_create(target_name.data(), target_name.length());
    nncapi->compile_session_create(target.get(), compile_options.get());
    compiler = nncapi->compile_session_get_compiler(compile_session.get());
#endif

    // fixed
    {
        ntt::tensor<float, ntt::fixed_shape<1, 16>> ta, tb, tc;
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::ops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }

    // ranked
    {
        auto shape = ntt::make_ranked_shape(1, 16);
        ntt::tensor<float, ntt::ranked_shape<2>> ta(shape), tb(shape),
            tc(shape);
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::ops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }

    // 2d binary
    {
      // pack and broadcast
        {
          ntt::tensor<float, ntt::fixed_shape<1, 16, 8>> ta;
          ntt::tensor<float, ntt::fixed_shape<8>> tb;
          std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
          std::fill(tb.elements().begin(), tb.elements().end(), 1.f);
          ntt::tensor<ntt::vector<float,4,4>, ntt::fixed_shape<1, 4, 2>> pa, pc;
          ntt::tensor<ntt::vector<float,4>, ntt::fixed_shape<2>> pb;
          ntt::pack<1,2>(ta,pa);
          ntt::pack<0>(tb,pb);
          ntt::binary<ntt::ops::add>(pa, pb, pc.view());
        } 
    }

    // 1
    {
        auto shape = ntt::make_ranked_shape(1);
        ntt::tensor<float, ntt::ranked_shape<1>> ta(shape), tb(shape),
            tc(shape);
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::sin>(ta, tb.view());
        assert(tb(0) == sinf(1.f));
        ntt::binary<ntt::ops::mul>(ta, tb, tc);
        assert(tc(0) == sinf(1.f));
    }

    // viewd tensor
    {
        ntt::tensor<float, ntt::fixed_shape<2, 3>> ta;
        ntt::tensor<float, ntt::fixed_shape<2, 1, 3>> tb;
        ntt::tensor_copy(ta.reshape(ntt::fixed_shape<2, 1, 3>{}), tb.view());
        assert(ta(0, 0) == tb(0, 0, 0));
        assert(ta(0, 1) == tb(0, 0, 1));
        assert(ta(0, 2) == tb(0, 0, 2));
        assert(ta(1, 0) == tb(1, 0, 0));
        assert(ta(1, 1) == tb(1, 0, 1));
        assert(ta(1, 2) == tb(1, 0, 2));
    }

    // fixed pack
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 16, 32>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::ranked_shape<tb.shape().rank()> inIndex;
            for (size_t i = 0; i < index.rank(); i++) {
                inIndex[i] = index[i];
            }
            NNCASE_UNUSED auto b = tb(index);
            auto start = index[1];
            for (size_t i = 0; i < 4; i++) {
                index[1] = start * 4 + i;
                NNCASE_UNUSED auto va = ta(index);
                NNCASE_UNUSED auto vb = b(ntt::ranked_shape<1>{i});
                assert(vb == va);
            }
        });
    }

    // fixed pack with pad
    {
        ntt::tensor<float, ntt::fixed_shape<1, 3, 4>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb);
        assert(tb(0, 0, 0)(0) == ta(0, 0, 0));
        assert(tb(0, 0, 0)(1) == ta(0, 1, 0));
        assert(tb(0, 0, 0)(2) == ta(0, 2, 0));
        assert(are_floats_equal(tb(0, 0, 0)(3), 0.f));

        ntt::tensor<float, ntt::fixed_shape<16>> tc;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<4>> td;
        std::iota(tc.elements().begin(), tc.elements().end(), 0.f);
        ntt::pack<0>(tc, td);
        for (size_t i = 0; i < 4; i++) {
            assert(td(ntt::ranked_shape<1>{i})(0) == tc(i * 4 + 0));
            assert(td(ntt::ranked_shape<1>{i})(1) == tc(i * 4 + 1));
            assert(td(ntt::ranked_shape<1>{i})(2) == tc(i * 4 + 2));
            assert(td(ntt::ranked_shape<1>{i})(3) == tc(i * 4 + 3));
        }
    }

    // fixed pack with pad, and unary
    {
        ntt::tensor<float, ntt::fixed_shape<1, 3, 4>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4>> tc;
        ntt::unary<ntt::ops::cos>(tb, tc);
        assert(tc(0, 0, 0)(0) == std::cos(ta(0, 0, 0)));
        assert(tc(0, 0, 0)(1) == std::cos(ta(0, 1, 0)));
        assert(tc(0, 0, 0)(2) == std::cos(ta(0, 2, 0)));
        assert(are_floats_equal(tc(0, 0, 0)(3), std::cos(0.0f)));
    }

    // pack(fixed_shape + fixed_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 8, 32>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::ranked_shape<tb.shape().rank()> inIndex;
            for (size_t i = 0; i < index.rank(); i++)
                inIndex[i] = index[i];
            auto b = tb(index);
            auto start = index[1];
            for (size_t i = 0; i < 8; i++) {
                index[1] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(ntt::ranked_shape<1>{i});
                if (va != vb) {
                    std::cerr << "va(" << va << ") != vb(" << vb << ")"
                              << std::endl;
                    std::abort();
                }
            }
        });
    }

    // pack(ranked_shape + ranked_shape)
    {
        auto a_shape = ntt::make_ranked_shape(1, 64, 32);
        auto b_shape = ntt::make_ranked_shape(1, 8, 32);
        ntt::tensor<float, ntt::ranked_shape<3>> ta(a_shape);
        ntt::tensor<ntt::vector<float, 8>, ntt::ranked_shape<3>> tb(b_shape);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::ranked_shape<rank> inIndex;
            for (size_t i = 0; i < index.rank(); i++)
                inIndex[i] = index[i];
            auto b = tb(index);
            auto start = index[1];
            for (size_t i = 0; i < 8; i++) {
                index[1] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(ntt::ranked_shape<1>{i});
                if (va != vb) {
                    std::cerr << "va(" << va << ") != vb(" << vb << ")"
                              << std::endl;
                    std::abort();
                }
            }
        });
    }

    // pack(fixed_shape + ranked_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;
        auto shape = ntt::make_ranked_shape(1, 8, 32);
        ntt::tensor<ntt::vector<float, 8>, ntt::ranked_shape<3>> tb(shape);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::ranked_shape<rank> inIndex;
            for (size_t i = 0; i < index.rank(); i++)
                inIndex[i] = index[i];
            auto b = tb(index);
            auto start = index[1];
            for (size_t i = 0; i < 8; i++) {
                index[1] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(ntt::ranked_shape<1>{i});
                if (va != vb) {
                    std::cerr << "va(" << va << ") != vb(" << vb << ")"
                              << std::endl;
                    std::abort();
                }
            }
        });
    }

    // pack(ranked_shape + fixed_shape)
    {
        auto shape = ntt::make_ranked_shape(1, 64, 32);
        ntt::tensor<float, ntt::ranked_shape<3>> ta(shape);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 8, 32>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::ranked_shape<rank> inIndex;
            for (size_t i = 0; i < index.rank(); i++)
                inIndex[i] = index[i];
            auto b = tb(index);
            auto start = index[1];
            for (size_t i = 0; i < 8; i++) {
                index[1] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(ntt::ranked_shape<1>{i});
                if (va != vb) {
                    std::cerr << "va(" << va << ") != vb(" << vb << ")"
                              << std::endl;
                    std::abort();
                }
            }
        });
    }

    // unpack(fixed_shape + fixed_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta, tc;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 16, 32>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::unpack<1>(tb, tc.view());
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // unpack(fixed_shape + ranked_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 16, 32>> tb;
        auto shape = ntt::make_ranked_shape(1, 64, 32);
        ntt::tensor<float, ntt::ranked_shape<3>> tc(shape);

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::unpack<1>(tb, tc.view());
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // vector unary
    {
        ntt::vector<float, 8> v1(1.f);
        NNCASE_UNUSED auto v2 = ntt::cos(v1);
        assert(v2(0) == std::cos(1.f));
    }

    // unpack(ranked_shape + fixed_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta, tc;
        auto shape = ntt::make_ranked_shape(1, 16, 32);
        ntt::tensor<ntt::vector<float, 4>, ntt::ranked_shape<3>> tb(shape);

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::unpack<1>(tb, tc.view());
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // unpack(ranked_shape + ranked_shape)
    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;

        auto shape1 = ntt::make_ranked_shape(1, 16, 32);
        ntt::tensor<ntt::vector<float, 4>, ntt::ranked_shape<3>> tb(shape1);

        auto shape2 = ntt::make_ranked_shape(1, 64, 32);
        ntt::tensor<float, ntt::ranked_shape<3>> tc(shape2);

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::unpack<1>(tb, tc.view());
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // im2col
    {
        ntt::tensor<float, ntt::fixed_shape<1, 1, 4, 4>> input;
        std::iota(input.elements().begin(), input.elements().end(), 0.f);
        ntt::tensor<float, ntt::fixed_shape<9, 16>> output;
        ntt::im2col(input, ntt::fixed_shape<3, 3>{}, ntt::fixed_shape<1, 1>{},
                    ntt::fixed_shape<1, 1, 1, 1>{}, ntt::fixed_shape<>{},
                    ntt::fixed_shape<>{}, output);
        // clang-format off
      assert(output(0,0) == 0.f); assert(output(0,1) == 0.f); assert(output(0,2) == 0.f); assert(output(0,3) == 0.f); assert(output(0,4) == 0.f); assert(output(0,5) == 0.f); assert(output(0,6) == 1.f); assert(output(0,7) == 2.f); assert(output(0,8) == 0.f); assert(output(0,9) == 4.f); assert(output(0,10) == 5.f); assert(output(0,11) == 6.f); assert(output(0,12) == 0.f); assert(output(0,13) == 8.f); assert(output(0,14) == 9.f); assert(output(0,15) == 10.f);
      assert(output(1,0) == 0.f); assert(output(1,1) == 0.f); assert(output(1,2) == 0.f); assert(output(1,3) == 0.f); assert(output(1,4) == 0.f); assert(output(1,5) == 1.f); assert(output(1,6) == 2.f); assert(output(1,7) == 3.f); assert(output(1,8) == 4.f); assert(output(1,9) == 5.f); assert(output(1,10) == 6.f); assert(output(1,11) == 7.f); assert(output(1,12) == 8.f); assert(output(1,13) == 9.f); assert(output(1,14) == 10.f); assert(output(1,15) == 11.f); 
      assert(output(2,0) == 0.f); assert(output(2,1) == 0.f); assert(output(2,2) == 0.f); assert(output(2,3) == 0.f); assert(output(2,4) == 1.f); assert(output(2,5) == 2.f); assert(output(2,6) == 3.f); assert(output(2,7) == 0.f); assert(output(2,8) == 5.f); assert(output(2,9) == 6.f); assert(output(2,10) == 7.f); assert(output(2,11) == 0.f); assert(output(2,12) == 9.f); assert(output(2,13) == 10.f); assert(output(2,14) == 11.f); assert(output(2,15) == 0.f); 
      assert(output(3,0) == 0.f); assert(output(3,1) == 0.f); assert(output(3,2) == 1.f); assert(output(3,3) == 2.f); assert(output(3,4) == 0.f); assert(output(3,5) == 4.f); assert(output(3,6) == 5.f); assert(output(3,7) == 6.f); assert(output(3,8) == 0.f); assert(output(3,9) == 8.f); assert(output(3,10) == 9.f); assert(output(3,11) == 10.f); assert(output(3,12) == 0.f); assert(output(3,13) == 12.f); assert(output(3,14) == 13.f); assert(output(3,15) == 14.f); 
      assert(output(4,0) == 0.f); assert(output(4,1) == 1.f); assert(output(4,2) == 2.f); assert(output(4,3) == 3.f); assert(output(4,4) == 4.f); assert(output(4,5) == 5.f); assert(output(4,6) == 6.f); assert(output(4,7) == 7.f); assert(output(4,8) == 8.f); assert(output(4,9) == 9.f); assert(output(4,10) == 10.f); assert(output(4,11) == 11.f); assert(output(4,12) == 12.f); assert(output(4,13) == 13.f); assert(output(4,14) == 14.f); assert(output(4,15) == 15.f); 
      assert(output(5,0) == 1.f); assert(output(5,1) == 2.f); assert(output(5,2) == 3.f); assert(output(5,3) == 0.f); assert(output(5,4) == 5.f); assert(output(5,5) == 6.f); assert(output(5,6) == 7.f); assert(output(5,7) == 0.f); assert(output(5,8) == 9.f); assert(output(5,9) == 10.f); assert(output(5,10) == 11.f); assert(output(5,11) == 0.f); assert(output(5,12) == 13.f); assert(output(5,13) == 14.f); assert(output(5,14) == 15.f); assert(output(5,15) == 0.f); 
      assert(output(6,0) == 0.f); assert(output(6,1) == 4.f); assert(output(6,2) == 5.f); assert(output(6,3) == 6.f); assert(output(6,4) == 0.f); assert(output(6,5) == 8.f); assert(output(6,6) == 9.f); assert(output(6,7) == 10.f); assert(output(6,8) == 0.f); assert(output(6,9) == 12.f); assert(output(6,10) == 13.f); assert(output(6,11) == 14.f); assert(output(6,12) == 0.f); assert(output(6,13) == 0.f); assert(output(6,14) == 0.f); assert(output(6,15) == 0.f); 
      assert(output(7,0) == 4.f); assert(output(7,1) == 5.f); assert(output(7,2) == 6.f); assert(output(7,3) == 7.f); assert(output(7,4) == 8.f); assert(output(7,5) == 9.f); assert(output(7,6) == 10.f); assert(output(7,7) == 11.f); assert(output(7,8) == 12.f); assert(output(7,9) == 13.f); assert(output(7,10) == 14.f); assert(output(7,11) == 15.f); assert(output(7,12) == 0.f); assert(output(7,13) == 0.f); assert(output(7,14) == 0.f); assert(output(7,15) == 0.f); 
      assert(output(8,0) == 5.f); assert(output(8,1) == 6.f); assert(output(8,2) == 7.f); assert(output(8,3) == 0.f); assert(output(8,4) == 9.f); assert(output(8,5) == 10.f); assert(output(8,6) == 11.f); assert(output(8,7) == 0.f); assert(output(8,8) == 13.f); assert(output(8,9) == 14.f); assert(output(8,10) == 15.f); assert(output(8,11) == 0.f); assert(output(8,12) == 0.f); assert(output(8,13) == 0.f); assert(output(8,14) == 0.f); assert(output(8,15) == 0.f);
        // clang-format on
    }

    // im2col packed on ic
    {
        ntt::tensor<float, ntt::fixed_shape<1, 4, 4, 4>> input;
        std::iota(input.elements().begin(), input.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4, 4>>
            packed_input;
        ntt::pack<1>(input, packed_input);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<9, 16>>
            packed_output;
        ntt::im2col(packed_input, ntt::fixed_shape<3, 3>{},
                    ntt::fixed_shape<1, 1>{}, ntt::fixed_shape<1, 1, 1, 1>{},
                    ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{},
                    packed_output);
        ntt::tensor<float, ntt::fixed_shape<36, 16>> unpacked_output;
        // packed [n,c/4,h,w,4] => [c/4 * h * w, b * oh * ow]
        // so unpack should after reshape
        ntt::unpack<0>(packed_output.reshape(ntt::fixed_shape<1, 9, 16>{}),
                       unpacked_output.reshape(ntt::fixed_shape<4, 9, 16>{}));
        ntt::tensor<float, ntt::fixed_shape<36, 16>> output;
        ntt::im2col(input, ntt::fixed_shape<3, 3>{}, ntt::fixed_shape<1, 1>{},
                    ntt::fixed_shape<1, 1, 1, 1>{}, ntt::fixed_shape<>{},
                    ntt::fixed_shape<>{}, output);
        ntt::apply(output.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = output(index);
            NNCASE_UNUSED auto c = unpacked_output(index);
            assert(a == c);
        });
    }

    // packed matmul 1d on k
    {
        ntt::tensor<float, ntt::fixed_shape<3, 16>> ta;
        ntt::tensor<float, ntt::fixed_shape<16, 2>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 2>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 2>> pa;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> pb;
        ntt::pack<1>(ta, pa);
        ntt::pack<0>(tb, pb);
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
        assert(tc(0, 0) == 2480.f);
        assert(tc(0, 1) == 2600.f);
        assert(tc(1, 0) == 6320.f);
        assert(tc(1, 1) == 6696.f);
        assert(tc(2, 0) == 10160.f);
        assert(tc(2, 1) == 10792.f);
    }

    // packed matmul 1d on m
    {
        ntt::tensor<float, ntt::fixed_shape<4, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 2>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 8>> pa;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 2>> pc;
        ntt::pack<0>(ta, pa);
        ntt::matmul<false>(pa, tb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{});
        assert(are_floats_equal(pc(0, 0)(0), 280.f));
        assert(are_floats_equal(pc(0, 1)(0), 308.f));
        assert(are_floats_equal(pc(0, 0)(1), 728.f));
        assert(are_floats_equal(pc(0, 1)(1), 820.f));
        assert(are_floats_equal(pc(0, 0)(2), 1176.f));
        assert(are_floats_equal(pc(0, 1)(2), 1332.f));
        assert(are_floats_equal(pc(0, 0)(3), 1624.f));
        assert(are_floats_equal(pc(0, 1)(3), 1844.f));
    }

    // packed matmul 1d on n
    {
        ntt::tensor<float, ntt::fixed_shape<3, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<8, 1>> pb;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<3, 1>> pc;
        ntt::pack<1>(tb, pb);
        ntt::matmul<false>(ta, pb, pc, ntt::fixed_shape<>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
        assert(are_floats_equal(pc(0, 0)(0), 560.f));
        assert(are_floats_equal(pc(0, 0)(1), 588.f));
        assert(are_floats_equal(pc(0, 0)(2), 616.f));
        assert(are_floats_equal(pc(0, 0)(3), 644.f));
        assert(are_floats_equal(pc(1, 0)(0), 1456.f));
        assert(are_floats_equal(pc(1, 0)(1), 1548.f));
        assert(are_floats_equal(pc(1, 0)(2), 1640.f));
        assert(are_floats_equal(pc(1, 0)(3), 1732.f));
        assert(are_floats_equal(pc(2, 0)(0), 2352.f));
        assert(are_floats_equal(pc(2, 0)(1), 2508.f));
        assert(are_floats_equal(pc(2, 0)(2), 2664.f));
        assert(are_floats_equal(pc(2, 0)(3), 2820.f));
    }

    // packed matmul 1d on m(A) and n(B)
    {
        ntt::tensor<float, ntt::fixed_shape<4, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 8>> pa;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<8, 1>> pb;
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<1, 1>> pc;
        ntt::pack<0>(ta, pa);
        ntt::pack<1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
        assert(are_floats_equal(pc(0, 0)(0, 0), 560.f));
        assert(are_floats_equal(pc(0, 0)(0, 1), 588.f));
        assert(are_floats_equal(pc(0, 0)(0, 2), 616.f));
        assert(are_floats_equal(pc(0, 0)(0, 3), 644.f));
        assert(are_floats_equal(pc(0, 0)(1, 0), 1456.f));
        assert(are_floats_equal(pc(0, 0)(1, 1), 1548.f));
        assert(are_floats_equal(pc(0, 0)(1, 2), 1640.f));
        assert(are_floats_equal(pc(0, 0)(1, 3), 1732.f));
        assert(are_floats_equal(pc(0, 0)(2, 0), 2352.f));
        assert(are_floats_equal(pc(0, 0)(2, 1), 2508.f));
        assert(are_floats_equal(pc(0, 0)(2, 2), 2664.f));
        assert(are_floats_equal(pc(0, 0)(2, 3), 2820.f));
        assert(are_floats_equal(pc(0, 0)(3, 0), 3248.f));
        assert(are_floats_equal(pc(0, 0)(3, 1), 3468.f));
        assert(are_floats_equal(pc(0, 0)(3, 2), 3688.f));
        assert(are_floats_equal(pc(0, 0)(3, 3), 3908.f));
    }

    // packed matmul 2d on mk(A) and k(B)
    {
        ntt::tensor<float, ntt::fixed_shape<4, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<1, 2>> pa;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<2, 4>> pb;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 4>> pc;
        ntt::pack<0, 1>(ta, pa);
        ntt::pack<0>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
        assert(are_floats_equal(pc(0, 0)(0), 560.f));
        assert(are_floats_equal(pc(0, 1)(0), 588.f));
        assert(are_floats_equal(pc(0, 2)(0), 616.f));
        assert(are_floats_equal(pc(0, 3)(0), 644.f));
        assert(are_floats_equal(pc(0, 0)(1), 1456.f));
        assert(are_floats_equal(pc(0, 1)(1), 1548.f));
        assert(are_floats_equal(pc(0, 2)(1), 1640.f));
        assert(are_floats_equal(pc(0, 3)(1), 1732.f));
        assert(are_floats_equal(pc(0, 0)(2), 2352.f));
        assert(are_floats_equal(pc(0, 1)(2), 2508.f));
        assert(are_floats_equal(pc(0, 2)(2), 2664.f));
        assert(are_floats_equal(pc(0, 3)(2), 2820.f));
        assert(are_floats_equal(pc(0, 0)(3), 3248.f));
        assert(are_floats_equal(pc(0, 1)(3), 3468.f));
        assert(are_floats_equal(pc(0, 2)(3), 3688.f));
        assert(are_floats_equal(pc(0, 3)(3), 3908.f));
    }

    // packed matmul 2d on k(A) and kn(B)
    {
        ntt::tensor<float, ntt::fixed_shape<4, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 4>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<4, 2>> pa;
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 1>> pb;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<4, 1>> pc;
        ntt::pack<1>(ta, pa);
        ntt::pack<0, 1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});
        assert(are_floats_equal(pc(0, 0)(0), 560.f));
        assert(are_floats_equal(pc(0, 0)(1), 588.f));
        assert(are_floats_equal(pc(0, 0)(2), 616.f));
        assert(are_floats_equal(pc(0, 0)(3), 644.f));
        assert(are_floats_equal(pc(1, 0)(0), 1456.f));
        assert(are_floats_equal(pc(1, 0)(1), 1548.f));
        assert(are_floats_equal(pc(1, 0)(2), 1640.f));
        assert(are_floats_equal(pc(1, 0)(3), 1732.f));
        assert(are_floats_equal(pc(2, 0)(0), 2352.f));
        assert(are_floats_equal(pc(2, 0)(1), 2508.f));
        assert(are_floats_equal(pc(2, 0)(2), 2664.f));
        assert(are_floats_equal(pc(2, 0)(3), 2820.f));
        assert(are_floats_equal(pc(3, 0)(0), 3248.f));
        assert(are_floats_equal(pc(3, 0)(1), 3468.f));
        assert(are_floats_equal(pc(3, 0)(2), 3688.f));
        assert(are_floats_equal(pc(3, 0)(3), 3908.f));
    }

    // packed matmul 2d on mk(A) and kn(B)
    {
        ntt::tensor<float, ntt::fixed_shape<4, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 4>> tb;
        ntt::tensor<float, ntt::fixed_shape<4, 4>> tc, unpackc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<1, 2>> pa;
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 1>> pb;
        ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<1, 1>> pc;
        ntt::pack<0, 1>(ta, pa);
        ntt::pack<0, 1>(tb, pb);
        ntt::matmul<false>(pa, pb, pc, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0, 1>{},
                           ntt::fixed_shape<0>{});
        ntt::unpack<0, 1>(pc, unpackc.view());
        ntt::matmul<false>(ta, tb, tc);
        ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
            assert(tc(index) == unpackc(index));
        });
    }

    // packed matmul 1d on k with broadcast
    {
        ntt::tensor<float, ntt::fixed_shape<1, 1, 3, 16>> ta;
        ntt::tensor<float, ntt::fixed_shape<2, 16, 4>> tb;
        ntt::tensor<float, ntt::fixed_shape<1, 2, 3, 4>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 1, 3, 2>> pa;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2, 4>> pb;
        ntt::pack<3>(ta, pa);
        ntt::pack<1>(tb, pb);
        ntt::matmul<false>(pa, pb, tc, ntt::fixed_shape<3>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{});
        assert(tc(0, 0, 0, 0) == 4960.f);
        assert(tc(0, 0, 0, 1) == 5080.f);
        assert(tc(0, 0, 0, 2) == 5200.f);
        assert(tc(0, 0, 0, 3) == 5320.f);
        assert(tc(0, 1, 0, 0) == 12640.f);
        assert(tc(0, 1, 0, 1) == 12760.f);
        assert(tc(0, 1, 0, 2) == 12880.f);
        assert(tc(0, 1, 0, 3) == 13000.f);
    }

    // norm matmul
    {
        ntt::tensor<float, ntt::fixed_shape<3, 4>> ta;
        ntt::tensor<float, ntt::fixed_shape<4, 2>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 2>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::matmul<false>(ta, tb, tc);
        assert(tc(0, 0) == 28.f);
        assert(tc(0, 1) == 34.f);
        assert(tc(1, 0) == 76.f);
        assert(tc(1, 1) == 98.f);
        assert(tc(2, 0) == 124.f);
        assert(tc(2, 1) == 162.f);
        ntt::tensor<float, ntt::fixed_shape<1, 1, 3, 4>> te;
        ntt::tensor<float, ntt::fixed_shape<2, 4, 5>> tf;
        std::iota(te.elements().begin(), te.elements().end(), 0.f);
        std::iota(tf.elements().begin(), tf.elements().end(), 0.f);
        ntt::tensor<float, ntt::fixed_shape<1, 2, 3, 5>> tg;
        ntt::matmul<false>(te, tf, tg);
        assert(tg(0, 0, 0, 0) == 70.f);
        assert(tg(0, 0, 1, 0) == 190.f);
        assert(tg(0, 0, 2, 0) == 310.f);
        assert(tg(0, 1, 0, 0) == 190.f);
        assert(tg(0, 1, 1, 0) == 630.f);
        assert(tg(0, 1, 2, 0) == 1070.f);
    }

    // transposeB matmul test
    {
        // 1. ref
        ntt::tensor<float, ntt::fixed_shape<8, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<8, 8>> tb;
        ntt::tensor<float, ntt::fixed_shape<8, 8>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::matmul<false>(ta, tb, tc);

        // transB norm
        {
            ntt::tensor<float, ntt::fixed_shape<8, 8>> tranb;
            ntt::transpose<ntt::fixed_shape<1, 0>>(tb, tranb);
            ntt::tensor<float, ntt::fixed_shape<8, 8>> tc1;
            ntt::matmul<false, false, true>(ta, tranb, tc1);
            ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
                assert(tc1(index) == tc(index));
            });

            // transB pack n
            {
                ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<2, 8>>
                    packb;
                ntt::pack<0>(tranb, packb);
                ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<8, 2>> tc2;
                ntt::matmul<false, false, true>(
                    ta, packb, tc2, ntt::fixed_shape<>{}, ntt::fixed_shape<>{},
                    ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});

                ntt::tensor<float, ntt::fixed_shape<8, 8>> tc2unpack;
                ntt::unpack<1>(tc2, tc2unpack);

                ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
                    assert(tc2unpack(index) == tc(index));
                });
            }
            // transB [M,K]<m> @ [N,K]<n>
            {
                ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<2, 8>>
                    packa;
                ntt::pack<0>(ta, packa);
                ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<2, 8>>
                    packb;
                ntt::pack<0>(tranb, packb);
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>> tc2;
                ntt::matmul<false, false, true>(
                    packa, packb, tc2, ntt::fixed_shape<0>{}, ntt::fixed_shape<>{},
                    ntt::fixed_shape<0>{}, ntt::fixed_shape<>{});

                ntt::tensor<float, ntt::fixed_shape<8, 8>> tc2unpack;
                ntt::unpack<0, 1>(tc2, tc2unpack);

                ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
                    assert(tc2unpack(index) == tc(index));
                });
            }

            // A[m,k]<m,k> @ B[n,k]<k,n>
            {
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    packb;
                ntt::pack<1, 0>(tranb, packb); // [n,k]<k,n>
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    packa;
                // note actully a should pack as [m,k]<k,m>
                ntt::pack<0, 1>(ta, packa); // [m,k]<m,k>
                // [m,n]<m,n>
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    tc2;
                ntt::matmul<false, false, true>(
                    packa, packb, tc2, ntt::fixed_shape<0, 1>{},
                    ntt::fixed_shape<>{}, ntt::fixed_shape<1, 0>{},
                    ntt::fixed_shape<>{});

                ntt::tensor<float, ntt::fixed_shape<8, 8>> tc2unpack;
                ntt::unpack<0, 1>(tc2, tc2unpack);

                ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
                    assert(tc2unpack(index) == tc(index));
                });
            }

            // A[m,k]<k,m> @ B[n,k]<k,n>
            {
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    packb;
                ntt::pack<1, 0>(tranb, packb); // [n,k]<k,n>
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    packa;
                ntt::pack<1, 0>(ta, packa); // [m,k]<k,m>
                // [m,n]<m,n>
                ntt::tensor<ntt::vector<float, 4, 4>, ntt::fixed_shape<2, 2>>
                    tc2;
                ntt::matmul<false, false, true>(
                    packa, packb, tc2, ntt::fixed_shape<1, 0>{},
                    ntt::fixed_shape<>{}, ntt::fixed_shape<1, 0>{},
                    ntt::fixed_shape<>{});

                ntt::tensor<float, ntt::fixed_shape<8, 8>> tc2unpack;
                ntt::unpack<0, 1>(tc2, tc2unpack);

                ntt::apply(tc.shape(), [&]([[maybe_unused]] auto index) {
                    assert(tc2unpack(index) == tc(index));
                });
            }
        }
    }

    // concat
    {
        ntt::tensor<float, ntt::fixed_shape<3, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<3, 16>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 24>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 1>> pa;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 2>> pb;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pc;
        ntt::pack<1>(ta, pa);
        ntt::pack<1>(tb, pb);
        ntt::concat<1>(std::make_tuple(pa, pb), pc);
        ntt::unpack<1>(pc, tc);

        assert(tc(0, 0) == 0.f);
        assert(tc(0, 1) == 1.f);
        assert(tc(0, 2) == 2.f);
        assert(tc(0, 3) == 3.f);
        assert(tc(0, 4) == 4.f);
        assert(tc(0, 5) == 5.f);
        assert(tc(0, 6) == 6.f);
        assert(tc(0, 7) == 7.f);
        assert(tc(0, 8) == 0.f);
        assert(tc(0, 9) == 1.f);
        assert(tc(0, 10) == 2.f);
        assert(tc(0, 11) == 3.f);
        assert(tc(0, 12) == 4.f);
        assert(tc(0, 13) == 5.f);
        assert(tc(0, 14) == 6.f);
        assert(tc(0, 15) == 7.f);
        assert(tc(0, 16) == 8.f);
        assert(tc(0, 17) == 9.f);
        assert(tc(0, 18) == 10.f);
        assert(tc(0, 19) == 11.f);
        assert(tc(0, 20) == 12.f);
        assert(tc(0, 21) == 13.f);
        assert(tc(0, 22) == 14.f);
        assert(tc(0, 23) == 15.f);
    }

    // slice
    {
        ntt::tensor<float, ntt::fixed_shape<3, 24>> ta;
        ntt::tensor<float, ntt::fixed_shape<3, 8>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 16>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::slice<ntt::fixed_shape<0>, ntt::fixed_shape<8>,
                   ntt::fixed_shape<1>, ntt::fixed_shape<1>>(ta, tb);
        ntt::slice<ntt::fixed_shape<8>, ntt::fixed_shape<24>,
                   ntt::fixed_shape<1>, ntt::fixed_shape<1>>(ta, tc);
        assert(tb(0, 0) == 0.f);
        assert(tb(0, 1) == 1.f);
        assert(tb(0, 2) == 2.f);
        assert(tb(0, 3) == 3.f);
        assert(tb(0, 4) == 4.f);
        assert(tb(0, 5) == 5.f);
        assert(tb(0, 6) == 6.f);
        assert(tb(0, 7) == 7.f);
        assert(tc(0, 0) == 8.f);
        assert(tc(0, 1) == 9.f);
        assert(tc(0, 2) == 10.f);
        assert(tc(0, 3) == 11.f);
        assert(tc(0, 4) == 12.f);
        assert(tc(0, 5) == 13.f);
        assert(tc(0, 6) == 14.f);
        assert(tc(0, 7) == 15.f);
    }

    // transpose
    {
        ntt::tensor<float, ntt::fixed_shape<3, 24>> ta;
        ntt::tensor<float, ntt::fixed_shape<24, 3>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::transpose<ntt::fixed_shape<1, 0>>(ta, tb);
        assert(tb(0, 0) == 0.0f);
        assert(tb(0, 1) == 24.f);
        assert(tb(0, 2) == 48.f);

        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pa;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pb;
        ntt::pack<1>(ta, pa);
        ntt::transpose<ntt::fixed_shape<1, 0>>(pa, pb);
        assert(pb(0, 0)(0) == 0.0f);
        assert(pb(0, 0)(1) == 1.0f);
        assert(pb(0, 0)(2) == 2.0f);
        assert(pb(0, 0)(3) == 3.0f);
        assert(pb(0, 1)(0) == 24.f);
        assert(pb(0, 1)(1) == 25.f);
        assert(pb(0, 1)(2) == 26.f);
        assert(pb(0, 1)(3) == 27.f);
        assert(pb(0, 2)(0) == 48.f);
        assert(pb(0, 2)(1) == 49.f);
        assert(pb(0, 2)(2) == 50.f);
        assert(pb(0, 2)(3) == 51.f);
    }

    // swish
    {
        ntt::tensor<float, ntt::fixed_shape<3, 24>> ta;
        ntt::tensor<float, ntt::fixed_shape<3, 24>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::unary<ntt::ops::swish>(ta, tb);

        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pa;
        ntt::pack<1>(ta, pa);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pb;
        ntt::unary<ntt::ops::swish>(pa, pb);
    }

    // swishb
    {
        ntt::tensor<float, ntt::fixed_shape<3, 24>> ta;
        ntt::tensor<float, ntt::fixed_shape<1>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 24>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 1.f);
        ntt::binary<ntt::ops::swishb>(ta, tb, tc);

        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pa;
        ntt::pack<1>(ta, pa);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pc;
        ntt::binary<ntt::ops::swishb>(pa, tb, pc);
    }

    // gather
    {
        ntt::tensor<float, ntt::fixed_shape<6, 3>> ta;
        ntt::tensor<size_t, ntt::fixed_shape<1, 3>> tb;
        ntt::tensor<size_t, ntt::fixed_shape<1, 3, 3>> tc;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().rbegin(), tb.elements().rend(), 0.f);
        ntt::gather<0>(ta, tb, tc);
        assert(tc(0, 2, 0) == 0.0f);
        assert(tc(0, 2, 1) == 1.0f);
        assert(tc(0, 2, 2) == 2.0f);
        assert(tc(0, 1, 0) == 3.0f);
        assert(tc(0, 1, 1) == 4.0f);
        assert(tc(0, 1, 2) == 5.0f);
        assert(tc(0, 0, 0) == 6.0f);
        assert(tc(0, 0, 1) == 7.0f);
        assert(tc(0, 0, 2) == 8.0f);

        ntt::tensor<float, ntt::fixed_shape<2, 3, 3>> td;
        ntt::tensor<size_t, ntt::fixed_shape<1, 2>> te;
        ntt::tensor<size_t, ntt::fixed_shape<2, 1, 2, 3>> tf;
        std::iota(td.elements().begin(), td.elements().end(), 0.f);
        std::iota(te.elements().rbegin(), te.elements().rend(), 0.f);
        ntt::gather<1>(td, te, tf);
        assert(tf(0, 0, 1, 0) == 0.0f);
        assert(tf(0, 0, 1, 1) == 1.0f);
        assert(tf(0, 0, 1, 2) == 2.0f);
        assert(tf(0, 0, 0, 0) == 3.0f);
        assert(tf(0, 0, 0, 1) == 4.0f);
        assert(tf(0, 0, 0, 2) == 5.0f);
    }

    // pad
    {
        ntt::tensor<float, ntt::fixed_shape<1, 2, 3>> td;
        ntt::tensor<float, ntt::fixed_shape<8, 2, 3>> te;
        std::iota(td.elements().begin(), td.elements().end(), 0.f);
        ntt::pad<0, 7, 0, 0, 0, 0>(td, te, 1.3f);
        assert(te(0, 0, 1) == 1.f);
        assert(te(1, 0, 1) == 1.3f);
        assert(te(2, 0, 1) == 1.3f);
        assert(te(3, 0, 1) == 1.3f);
    }

    {
        ntt::tensor<float, ntt::fixed_shape<2, 8>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<2, 2>> tav;
        std::fill(ta.elements().begin(), ta.elements().begin() + 8, 1.f);
        std::fill(ta.elements().begin() + 8, ta.elements().end(), 3.2f);
        ntt::pack<1>(ta, tav.view());

        ntt::tensor<float, ntt::fixed_shape<2, 1>> tb;
        ntt::reduce_sum<ntt::fixed_shape<1>, ntt::fixed_shape<1>>(tav, tb);
        assert(are_floats_equal(tb(0, 0), 8.f));
        assert(are_floats_equal(tb(1, 0), 25.6f));
    }

    {
        ntt::tensor<float, ntt::fixed_shape<1, 16>> ta;
        ntt::tensor<int32_t, ntt::fixed_shape<1, 16>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::cast(ta, tb.view());
        assert(tb(0, 0) == 0);
        assert(tb(0, 2) == 2);
    }

    {
        ntt::tensor<float, ntt::fixed_shape<1, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 16, 32>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::tensor<ntt::vector<int32_t, 4>, ntt::fixed_shape<1, 16, 32>> tc;
        ntt::cast(tb, tc);
        assert(tc(0, 0, 0)(0) == 0);
        assert(tc(0, 0, 0)(1) == 32);
        assert(tc(0, 0, 0)(2) == 64);
    }

    {
        ntt::tensor<float, ntt::fixed_shape<1, 2>> ta;
        ntt::tensor<float, ntt::fixed_shape<2, 2>> tb;
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::expand(ta, tb.view());
        assert(are_floats_equal(tb(0, 0), 0.f));
        assert(are_floats_equal(tb(0, 1), 1.f));
        assert(are_floats_equal(tb(1, 0), 0.f));
        assert(are_floats_equal(tb(1, 1), 1.f));
    }

    {
        ntt::tensor<bool, ntt::fixed_shape<2, 2>> tcond;
        ntt::tensor<float, ntt::fixed_shape<2, 2>> tx;
        ntt::tensor<float, ntt::fixed_shape<2, 2>> ty;
        ntt::tensor<float, ntt::fixed_shape<2, 2>> tout;
        tcond(0, 0) = true;
        tcond(0, 1) = false;
        tcond(1, 0) = false;
        tcond(1, 1) = true;
        std::iota(tx.elements().begin(), tx.elements().end(), 0.f);
        std::iota(ty.elements().begin(), ty.elements().end(), 4.f);
        ntt::where(tcond, tx, ty, tout.view());
        assert(are_floats_equal(tout(0, 0), 0.f));
        assert(are_floats_equal(tout(0, 1), 5.f));
        assert(are_floats_equal(tout(1, 0), 6.f));
        assert(are_floats_equal(tout(1, 1), 3.f));
    }

    {
        ntt::tensor<float, ntt::fixed_shape<2, 4>> ta;
        ta(0, 0) = 0.f;
        ta(0, 1) = 2.f;
        ta(0, 2) = 4.f;
        ta(0, 3) = 6.f;
        ta(1, 0) = 7.f;
        ta(1, 1) = 5.f;
        ta(1, 2) = 3.f;
        ta(1, 3) = 7.f;

        ntt::tensor<int64_t, ntt::fixed_shape<2, 1>> tb;
        ntt::reduce_arg<ntt::ops::max, 1, false, true>(
            ta, tb.view(), ntt::fixed_shape<>(), ntt::fixed_shape<>());
        assert(tb(0, 0) == 3);
        assert(tb(1, 0) == 0);

        ntt::tensor<int64_t, ntt::fixed_shape<1, 4>> tc;
        ntt::reduce_arg<ntt::ops::max, 0, false, true>(
            ta, tc.view(), ntt::fixed_shape<>(), ntt::fixed_shape<>());
        assert(tc(0, 0) == 1);
        assert(tc(0, 1) == 1);
        assert(tc(0, 2) == 0);
        assert(tc(0, 3) == 1);

        ntt::tensor<int64_t, ntt::fixed_shape<2>> td;
        ntt::reduce_arg<ntt::ops::max, 1, true, false>(
            ta, td.view(), ntt::fixed_shape<>(), ntt::fixed_shape<>());
        assert(td(0) == 3);
        assert(td(1) == 3);
    }

    {
        ntt::tensor<float, ntt::fixed_shape<1, 16>> ta, tb, tc;
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::erf>(ta, tb.view());
        assert(are_floats_equal(tb(0, 0), erf(1.f)));
    }

#if 0
    auto kmodel = read_file(
        R"(/mnt/home-nas/work/repo/nncase/tests_output/UnitTestCPUTarget/TestSimpleUnary/TestSimpleUnary.kmodel)");

    interpreter *interp;
    TRY(nncase_interp_create(&interp));
    TRY(nncase_interp_load_model(interp, kmodel.data(), kmodel.size(), false));

    runtime_function *entry;
    TRY(nncase_interp_get_entry_func(interp, &entry));

    buffer_allocator *host_alloc;
    TRY(nncase_buffer_allocator_get_host(&host_alloc));

    datatype_node *dtype_int64, *dtype_float32;
    TRY(nncase_dtype_create_prime(dt_int64, &dtype_int64));
    TRY(nncase_dtype_create_prime(dt_float32, &dtype_float32));

    float x[] = {-1.f};
    buffer_node *x_buf;
    TRY(nncase_buffer_allocator_alloc(host_alloc, sizeof(x), nullptr, &x_buf));
    {
        host_buffer_node *x_host_buf;
        void *x_buf_data;
        TRY(nncase_buffer_as_host(x_buf, &x_host_buf));
        TRY(nncase_host_buffer_map(x_host_buf, map_write, &x_buf_data,
                                   nullptr));
        memcpy(x_buf_data, x, sizeof(x));
        TRY(nncase_host_buffer_unmap(x_host_buf));
        TRY(nncase_object_release((object_node *)x_host_buf));
    }

    tensor_node *x_tensor;
    uint32_t dims[] = {1, 1};
    uint32_t strides[] = {1, 1};
    nncase_buffer_slice x_buffer_slice{x_buf, 0, sizeof(x)};
    TRY(nncase_tensor_create(dtype_float32, dims, 1, strides, 1,
                             &x_buffer_slice, &x_tensor));

    value_node *params[] = {(value_node *)x_tensor};
    tensor_node *ret = nullptr;

    auto time_begin = std::chrono::steady_clock::now();

    TRY(nncase_func_invoke(entry, params, 1, (value_node **)&ret));

    auto time_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        time_end - time_begin);
    printf("Duration: %.2fms\n", duration.count() / 1e3);

    uint32_t ret_dims_len;
    TRY(nncase_tensor_get_dims(ret, nullptr, &ret_dims_len));
    std::vector<uint32_t> ret_dims(ret_dims_len);
    TRY(nncase_tensor_get_dims(ret, ret_dims.data(), &ret_dims_len));

    nncase_buffer_slice out_buffer_slice;
    TRY(nncase_tensor_get_buffer(ret, &out_buffer_slice));
    {
        host_buffer_node *ret_host_buf;
        void *ret_buf_data;
        uint32_t ret_bytes;
        TRY(nncase_buffer_as_host(out_buffer_slice.buffer, &ret_host_buf));
        TRY(nncase_host_buffer_map(ret_host_buf, map_read, &ret_buf_data,
                                   &ret_bytes));

        auto ret_float_data = (float *)ret_buf_data;
        std::cout << *ret_float_data << std::endl;

        TRY(nncase_host_buffer_unmap(ret_host_buf));
        TRY(nncase_object_release((object_node *)ret_host_buf));
    }

    TRY(nncase_object_release((object_node *)out_buffer_slice.buffer));
    TRY(nncase_object_release((object_node *)ret));
    TRY(nncase_object_release((object_node *)x_buf));
    TRY(nncase_object_release((object_node *)x_tensor));
    TRY(nncase_object_release((object_node *)dtype_int64));
    TRY(nncase_interp_free(interp));
#endif
    return 0;
}
