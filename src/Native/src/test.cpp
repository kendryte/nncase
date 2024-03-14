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
        std::fill(ta.buffer().begin(), ta.buffer().end(), 1.f);
        ntt::unary<ntt::mathops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::mathops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }

    // ranked
    {
        auto shape = ntt::make_ranked_shape(1, 16);
        ntt::tensor<float, ntt::ranked_shape<2>> ta(shape), tb(shape),
            tc(shape);
        std::fill(ta.buffer().begin(), ta.buffer().end(), 1.f);
        ntt::unary<ntt::mathops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::mathops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }

    // 1
    {
        auto shape = ntt::make_ranked_shape(1);
        ntt::tensor<float, ntt::ranked_shape<1>> ta(shape), tb(shape),
            tc(shape);
        std::fill(ta.buffer().begin(), ta.buffer().end(), 1.f);
        ntt::unary<ntt::mathops::sin>(ta, tb.view());
        assert(tb(0) == sinf(1.f));
        ntt::binary<ntt::mathops::mul>(ta, tb, tc);
        assert(tc(0) == sinf(1.f));
    }

    // viewd tensor
    // {
    //     ntt::tensor<float, ntt::fixed_shape<2, 3>> ta;
    //     ntt::tensor<float, ntt::fixed_shape<2, 1, 3>> tb;
    //     ntt::tensor_copy(ta, tb.view());
    //     assert(ta(0, 0) == tb(0, 0, 0));
    //     assert(ta(0, 1) == tb(0, 0, 1));
    //     assert(ta(0, 2) == tb(0, 0, 2));
    //     assert(ta(1, 0) == tb(1, 0, 0));
    //     assert(ta(1, 1) == tb(1, 0, 1));
    //     assert(ta(1, 2) == tb(1, 0, 2));
    // }

    // fixed pack
    {
        ntt::tensor<float, ntt::fixed_shape<16, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<16, 16, 32>> tb;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
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
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        ntt::pack<1>(ta, tb);
        assert(tb(0, 0, 0)(0) == ta(0, 0, 0));
        assert(tb(0, 0, 0)(1) == ta(0, 1, 0));
        assert(tb(0, 0, 0)(2) == ta(0, 2, 0));
        assert(tb(0, 0, 0)(3) == 0.f);
    }

    // fixed pack with pad, and unary
    {
        ntt::tensor<float, ntt::fixed_shape<1, 3, 4>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4>> tb;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<1, 1, 4>> tc;
        ntt::unary<ntt::mathops::cos>(tb, tc);
        assert(tc(0, 0, 0)(0) == std::cos(ta(0, 0, 0)));
        assert(tc(0, 0, 0)(1) == std::cos(ta(0, 1, 0)));
        assert(tc(0, 0, 0)(2) == std::cos(ta(0, 2, 0)));
        assert(tc(0, 0, 0)(3) == std::cos(0.0f));
    }

    // fixed unpack
    {
        ntt::tensor<float, ntt::fixed_shape<16, 64, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<16, 16, 32>> tb;
        ntt::tensor<float, ntt::fixed_shape<16, 64, 32>> tc;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        ntt::pack<1>(ta, tb.view());
        ntt::unpack<1>(tb, tc.view());
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // fixed unpack with pad
    {
        ntt::tensor<float, ntt::fixed_shape<16, 62, 32>> ta;
        ntt::tensor<ntt::vector<float, 4>, ntt::fixed_shape<16, 16, 32>> tb;
        ntt::tensor<float, ntt::fixed_shape<16, 62, 32>> tc;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        ntt::pack<1>(ta, tb);
        ntt::unpack<1>(tb, tc);
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }
    // layer_norm
    {
        ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_1;
        ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_4;
        ntt::tensor<float, ntt::fixed_shape<16, 2>> buffer_7;
        std::iota(buffer_1.buffer().begin(), buffer_1.buffer().end(), 0.f);
        std::iota(buffer_4.buffer().begin(), buffer_4.buffer().end(), 0.f);
        std::iota(buffer_7.buffer().begin(), buffer_7.buffer().end(), 0.f);

        // no pack
        ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_11;
        packed_layer_norm<1>(buffer_1, buffer_4, buffer_7, buffer_11, 1e-06,
                             true, ntt::fixed_shape<>{}, ntt::fixed_shape<>{});
        assert(buffer_11(0, 0, 0) == 0.0f);
        assert(std::abs(buffer_11(0, 0, 1) - (-0.57043804f)) < 1e-4f);
        assert(std::abs(buffer_11(0, 1, 0) - (-0.92426393f)) < 1e-4f);
        assert(std::abs(buffer_11(0, 1, 1) - (-1.06147768f)) < 1e-4f);
        assert(std::abs(buffer_11(0, 15, 0) - (77.11314114f)) < 1e-4f);
        assert(std::abs(buffer_11(0, 15, 1) - (83.04106739f)) < 1e-4f);

        // packed
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_2;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> buffer_5;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> buffer_8;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_9;
        pack<1>(buffer_1, buffer_2);
        pack<0>(buffer_4, buffer_5);
        pack<0>(buffer_7, buffer_8);
        packed_layer_norm<1>(buffer_2, buffer_5, buffer_8, buffer_9,
                             ntt::vector<float, 8>{1E-06}, true,
                             ntt::fixed_shape<1>{}, ntt::fixed_shape<0>{});

        ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_10;
        unpack<1>(buffer_9, buffer_10);
        assert(buffer_10(0, 0, 0) == 0.0f);
        assert(std::abs(buffer_10(0, 0, 1) - (-0.57043804f)) < 1e-4f);
        assert(std::abs(buffer_10(0, 1, 0) - (-0.92426393f)) < 1e-4f);
        assert(std::abs(buffer_10(0, 1, 1) - (-1.06147768f)) < 1e-4f);
        assert(std::abs(buffer_10(0, 15, 0) - (77.11314114f)) < 1e-4f);
        assert(std::abs(buffer_10(0, 15, 1) - (83.04106739f)) < 1e-4f);
    }

    // soft max
    {
        ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_1;
        std::iota(buffer_1.buffer().begin(), buffer_1.buffer().end(), 0.f);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_2;

        pack<1>(buffer_1, buffer_2);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<1, 2, 2>> buffer_9;
        packed_softmax<1>(buffer_2, buffer_9, ntt::fixed_shape<1>{});
        ntt::tensor<float, ntt::fixed_shape<1, 16, 2>> buffer_10;
        unpack<1>(buffer_9, buffer_10);

        assert(std::abs(buffer_10(0, 13, 0) - (1.58368867e-02)) < 1e-6f);
        assert(std::abs(buffer_10(0, 13, 1) - (1.58368867e-02)) < 1e-6f);
        assert(std::abs(buffer_10(0, 14, 0) - (1.17019644e-01)) < 1e-6f);
        assert(std::abs(buffer_10(0, 14, 1) - (1.17019644e-01)) < 1e-6f);
        assert(std::abs(buffer_10(0, 15, 0) - (8.64664717e-01)) < 1e-6f);
        assert(std::abs(buffer_10(0, 15, 1) - (8.64664717e-01)) < 1e-6f);
    }

    // packed matmul 1d on k
    {
        ntt::tensor<float, ntt::fixed_shape<3, 16>> ta;
        ntt::tensor<float, ntt::fixed_shape<16, 2>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 2>> tc;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        std::iota(tb.buffer().begin(), tb.buffer().end(), 0.f);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 2>> pa;
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<2, 2>> pb;
        ntt::pack<1>(ta, pa);
        ntt::pack<0>(tb, pb);
        ntt::packed_matmul(pa, pb, tc, ntt::fixed_shape<1>{},
                           ntt::fixed_shape<0>{}, ntt::fixed_shape<0>{},
                           ntt::fixed_shape<0>{});
        assert(tc(0, 0) == 2480.f);
        assert(tc(0, 1) == 2600.f);
        assert(tc(1, 0) == 6320.f);
        assert(tc(1, 1) == 6696.f);
        assert(tc(2, 0) == 10160.f);
        assert(tc(2, 1) == 10792.f);
    }

    // norm matmul
    {
        ntt::tensor<float, ntt::fixed_shape<3, 4>> ta;
        ntt::tensor<float, ntt::fixed_shape<4, 2>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 2>> tc;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        std::iota(tb.buffer().begin(), tb.buffer().end(), 0.f);
        ntt::matmul(ta, tb, tc);
        assert(tc(0, 0) == 28.f);
        assert(tc(0, 1) == 34.f);
        assert(tc(1, 0) == 76.f);
        assert(tc(1, 1) == 98.f);
        assert(tc(2, 0) == 124.f);
        assert(tc(2, 1) == 162.f);
    }

    // concat
    {
        ntt::tensor<float, ntt::fixed_shape<3, 8>> ta;
        ntt::tensor<float, ntt::fixed_shape<3, 16>> tb;
        ntt::tensor<float, ntt::fixed_shape<3, 24>> tc;
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        std::iota(tb.buffer().begin(), tb.buffer().end(), 0.f);
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
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
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
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
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
        std::iota(ta.buffer().begin(), ta.buffer().end(), 0.f);
        ntt::unary<ntt::mathops::swish>(ta, tb);

        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pa;
        ntt::pack<1>(ta, pa);
        ntt::tensor<ntt::vector<float, 8>, ntt::fixed_shape<3, 3>> pb;
        ntt::unary<ntt::mathops::swish>(pa, pb);
    }

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
    return 0;
}
