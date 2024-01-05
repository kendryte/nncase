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
