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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <nncase/api.h>
#include <nncase/compiler.h>
#include <nncase/io_utils.h>
// #include <nncase/ntt/caching.h>
#include <nncase/ntt/ntt.h>
#include <nncase/runtime/runtime_tensor.h>
#include <string_view>
#include <sys/stat.h>
#include <type_traits>

namespace nncase::ntt::runtime {
// just for test
cpu_thread_context_t &cpu_thread_context_t::current() noexcept {
    static cpu_thread_context_t ctx{.tid = 0, .bid = 0, .cid = 0};
    return ctx;
}
} // namespace nncase::ntt::runtime

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

void test_shape() {

    // fixed shape
    {
        auto shape = ntt::fixed_shape_v<1, 16>;
        auto dim1 = shape[dim_zero];
        static_assert(dim1.value == 1);
        auto dim2 = shape[dim_one];
        static_assert(dim2.value == 16);
        auto sub_dim = dim2 - shape.rank();
        static_assert(sub_dim.value == 14);
        static_assert(FixedDimension<decltype(sub_dim)>);
        static_assert(shape.contains(1));
        static_assert(shape.contains((size_t)16));
        static_assert(!shape.contains(2));

        auto appended_shape = shape.append(fixed_dim_v<2>);
        static_assert(appended_shape.rank() == 3);
        static_assert(appended_shape[dim_zero] == 1);
        static_assert(appended_shape[dim_one] == 16);
        static_assert(appended_shape[2] == 2);

        auto squeezed_shape = ntt::squeeze_dims(shape, fixed_shape_v<0>);
        static_assert(squeezed_shape.rank() == 1);

        auto concat_shape = shape.concat(fixed_shape_v<2, 3>);
        static_assert(concat_shape.rank() == 4);

        auto replaced_shape = shape.replace_at<0>(fixed_dim_v<2>);
        static_assert(replaced_shape.rank() == 2);
        static_assert(replaced_shape.length() == 32);
        static_assert(
            linear_size(replaced_shape, default_strides(replaced_shape)) == 32);
    }

    {
        float arr[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        auto buffer = std::span(arr);
        auto tv = ntt::make_tensor_view(buffer, ntt::fixed_shape_v<2, 4>);
        static_assert(tv.rank() == 2);
    }

    {
        const float arr[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        auto buffer = std::span(arr);
        auto tv = ntt::make_tensor_view(buffer, ntt::fixed_shape_v<2, 4>);
        static_assert(tv.rank() == 2);
    }

    {
        const float buffer[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        auto tv = ntt::make_tensor_view(buffer, ntt::fixed_shape_v<2, 4>);
        static_assert(tv.rank() == 2);
    }

    {
        using v1_type = ntt::vector<float, 8>;
        using v2_type = ntt::replace_element_t<v1_type, int>;
        static_assert(std::is_same_v<v2_type, ntt::vector<int, 8>>);
    }
}

void test_sharding() {
    // local_index
    {
        using mesh_type =
            ntt::distributed::mesh<ntt::distributed::topology::thread, 1>;

        static_assert(ntt::distributed::program_dim<
                          ntt::distributed::topology::thread>() == 1);
        static_assert(
            ntt::distributed::detail::get_submesh_end<mesh_type,
                                                      topology::thread>() == 1);
        static_assert(ntt::distributed::detail::get_submesh_rank<
                          mesh_type, topology::thread>() == 1);
        static_assert(
            ntt::distributed::detail::get_submesh_rank<mesh_type,
                                                       topology::chip>() == 0);
        static_assert(ntt::distributed::detail::get_submesh_start<
                          mesh_type, topology::thread>() == 0);
        auto program_ids = ntt::distributed::program_ids<>();
        auto local_index = mesh_type::index_from_program_ids(program_ids);
        static_assert(local_index.rank() == 1);

        auto sharding = ntt::distributed::make_sharding<mesh_type>(
            ntt::distributed::shard_policy::B,
            ntt::distributed::shard_policy::B);

        auto global_shape = ntt::fixed_shape_v<2, 4>;
        constexpr auto local_shape =
            sharding.shard_shape(global_shape, local_index);
        static_assert(local_shape == ntt::fixed_shape_v<2, 4>);

        const float buffer[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        auto s_tensor = ntt::distributed::make_sharded_tensor_view(
            buffer, global_shape,
            ntt::distributed::make_sharding<mesh_type>(
                ntt::distributed::shard_policy::B,
                ntt::distributed::shard_policy::B),
            ntt::fixed_strides_v<4, 1>);
        static_assert(s_tensor.local().shape() == ntt::fixed_shape_v<2, 4>);
    }

    // Sharding
    {
        using mesh_type =
            ntt::distributed::mesh<ntt::distributed::topology::thread, 1, 1, 1>;

        auto sharding = ntt::distributed::make_sharding<mesh_type>(
            ntt::distributed::shard_policy::B,
            ntt::distributed::shard_policy::S<2>(),
            ntt::distributed::shard_policy::B);
        using sharding_type = std::remove_cv_t<decltype(sharding)>;
        static_assert(
            ntt::distributed::detail::mesh_axes_mask_of_split_shard_policies<
                sharding_type>() == ntt::fixed_shape_v<0, 0, 1>);
        static_assert(
            ntt::distributed::detail::mesh_axes_of_non_split_shard_policies<
                sharding_type>() == ntt::fixed_shape_v<0, 1>);
    }
}

void test_matmul_normal() {
    // no pack
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 4>);
        auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<4, 2>);
        auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 2>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
        ntt::matmul(ta, tb, tc);
        assert(tc(0, 0) == 28.f);
        assert(tc(0, 1) == 34.f);
        assert(tc(1, 0) == 76.f);
        assert(tc(1, 1) == 98.f);
        assert(tc(2, 0) == 124.f);
        assert(tc(2, 1) == 162.f);
        auto te = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 4>);
        auto tf = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 4, 5>);
        std::iota(te.elements().begin(), te.elements().end(), 0.f);
        std::iota(tf.elements().begin(), tf.elements().end(), 0.f);
        auto tg = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 3, 5>);
        ntt::matmul(te, tf, tg);
        assert(tg(0, 0, 0, 0) == 70.f);
        assert(tg(0, 0, 1, 0) == 190.f);
        assert(tg(0, 0, 2, 0) == 310.f);
        assert(tg(0, 1, 0, 0) == 190.f);
        assert(tg(0, 1, 1, 0) == 630.f);
        assert(tg(0, 1, 2, 0) == 1070.f);
    }

    // fixed
    {
        auto shape = ntt::fixed_shape_v<1, 16>;
        auto ta = ntt::make_tensor<float>(shape);
        auto tb = ntt::make_tensor<float>(shape);
        auto tc = ntt::make_tensor<float>(shape);
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::ops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }

    // ranked
    {
        auto shape = ntt::make_shape(1, 16);
        auto ta = ntt::make_tensor<float>(shape);
        auto tb = ntt::make_tensor<float>(shape);
        auto tc = ntt::make_tensor<float>(shape);
        std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
        ntt::unary<ntt::ops::sin>(ta, tb.view());
        assert(tb(0, 0) == sinf(1.f));
        ntt::binary<ntt::ops::mul>(ta, tb, tc);
        assert(tc(0, 0) == sinf(1.f));
    }
}

void test_pack() {
    // fixed pack
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 16, 32>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::dynamic_shape_t<tb.shape().rank()> inIndex;
            ntt::loop<inIndex.rank()>([&](auto &i) { inIndex[i] = index[i]; });
            NNCASE_UNUSED auto b = tb(index);
            auto start = index[1_dim];
            for (ntt::dim_t i = 0; i < 4; i++) {
                index[1_dim] = start * 4 + i;
                NNCASE_UNUSED auto va = ta(index);
                NNCASE_UNUSED auto vb = b(i);
                assert(vb == va);
            }
        });
    }

    // fixed pack with pad
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 4>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 4>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb, fixed_shape_v<1>);
        assert(tb(0, 0, 0)(0) == ta(0, 0, 0));
        assert(tb(0, 0, 0)(1) == ta(0, 1, 0));
        assert(tb(0, 0, 0)(2) == ta(0, 2, 0));
        assert(are_floats_equal(tb(0, 0, 0)(3), 0.f));

        auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<16>);
        auto td =
            ntt::make_tensor<ntt::vector<float, 4>>(ntt::fixed_shape_v<4>);
        std::iota(tc.elements().begin(), tc.elements().end(), 0.f);
        ntt::pack(tc, td, fixed_shape_v<0>);
        for (ntt::dim_t i = 0; i < 4; i++) {
            assert(td(i)(0) == tc(i * 4 + 0));
            assert(td(i)(1) == tc(i * 4 + 1));
            assert(td(i)(2) == tc(i * 4 + 2));
            assert(td(i)(3) == tc(i * 4 + 3));
        }
    }

    // fixed pack with pad, and unary
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 4>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 4>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb, fixed_shape_v<1>);
        auto tc = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 4>);
        ntt::unary<ntt::ops::cos>(tb, tc);
        assert(tc(0, 0, 0)(0) == std::cos(ta(0, 0, 0)));
        assert(tc(0, 0, 0)(1) == std::cos(ta(0, 1, 0)));
        assert(tc(0, 0, 0)(2) == std::cos(ta(0, 2, 0)));
        assert(are_floats_equal(tc(0, 0, 0)(3), std::cos(0.0f)));
    }

    // pack(fixed_shape + fixed_shape)
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb = ntt::make_tensor<ntt::vector<float, 8>>(
            ntt::fixed_shape_v<1, 8, 32>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::dynamic_shape_t<tb.shape().rank()> inIndex;
            ntt::loop<inIndex.rank()>([&](auto &i) { inIndex[i] = index[i]; });
            auto b = tb(index);
            auto start = index[1_dim];
            for (ntt::dim_t i = 0; i < 8; i++) {
                index[1_dim] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(i);
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
        auto ta = ntt::make_tensor<float>(ntt::make_shape(1, 64, 32));
        auto tb =
            ntt::make_tensor<ntt::vector<float, 8>>(ntt::make_shape(1, 8, 32));
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::dynamic_shape_t<rank> inIndex;
            ntt::loop<inIndex.rank()>([&](auto &i) { inIndex[i] = index[i]; });
            auto b = tb(index);
            auto start = index[1_dim];
            for (ntt::dim_t i = 0; i < 8; i++) {
                index[1_dim] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(i);
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
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb =
            ntt::make_tensor<ntt::vector<float, 8>>(ntt::make_shape(1, 8, 32));
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::dynamic_shape_t<rank> inIndex;
            ntt::loop<inIndex.rank()>([&](auto &i) { inIndex[i] = index[i]; });
            auto b = tb(index);
            auto start = index[1_dim];
            for (size_t i = 0; i < 8; i++) {
                index[1_dim] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(i);
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
        auto ta = ntt::make_tensor<float>(ntt::make_shape(1, 64, 32));
        auto tb = ntt::make_tensor<ntt::vector<float, 8>>(
            ntt::fixed_shape_v<1, 8, 32>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        constexpr auto rank = tb.shape().rank();
        ntt::apply(tb.shape(), [&](auto index) {
            ntt::dynamic_shape_t<rank> inIndex;
            ntt::loop<inIndex.rank()>([&](auto &i) { inIndex[i] = index[i]; });
            auto b = tb(index);
            auto start = index[1_dim];
            for (size_t i = 0; i < 8; i++) {
                index[1_dim] = start * 8 + i;
                auto va = ta(index);
                auto vb = b(i);
                if (va != vb) {
                    std::cerr << "va(" << va << ") != vb(" << vb << ")"
                              << std::endl;
                    std::abort();
                }
            }
        });
    }

    // 2d binary
    { // pack and broadcast
        {
            auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 16, 8>);
            auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<8>);
            std::fill(ta.elements().begin(), ta.elements().end(), 1.f);
            std::fill(tb.elements().begin(), tb.elements().end(), 1.f);
            auto pa = ntt::make_tensor<ntt::vector<float, 4, 4>>(
                ntt::fixed_shape_v<1, 4, 2>);
            auto pc = ntt::make_tensor<ntt::vector<float, 4, 4>>(
                ntt::fixed_shape_v<1, 4, 2>);
            auto pb =
                ntt::make_tensor<ntt::vector<float, 4>>(ntt::fixed_shape_v<2>);
            ntt::pack(ta, pa, fixed_shape_v<1, 2>);
            ntt::pack(tb, pb, fixed_shape_v<0>);
            ntt::binary<ntt::ops::add>(pa, pb, pc.view());
        }
    }
    // unpack(fixed_shape + fixed_shape)
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 16, 32>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::unpack(tb, tc.view(), fixed_shape_v<1>);
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }
    // unpack(fixed_shape + ranked_shape)
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 16, 32>);
        auto tc = ntt::make_tensor<float>(ntt::make_shape(1, 64, 32));

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::unpack(tb, tc.view(), fixed_shape_v<1>);
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
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb =
            ntt::make_tensor<ntt::vector<float, 4>>(ntt::make_shape(1, 16, 32));

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::unpack(tb, tc.view(), fixed_shape_v<1>);
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }

    // unpack(ranked_shape + ranked_shape)
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb =
            ntt::make_tensor<ntt::vector<float, 4>>(ntt::make_shape(1, 16, 32));
        auto tc = ntt::make_tensor<float>(ntt::make_shape(1, 64, 32));

        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), fixed_shape_v<1>);
        ntt::unpack(tb, tc.view(), fixed_shape_v<1>);
        ntt::apply(tc.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = ta(index);
            NNCASE_UNUSED auto c = tc(index);
            assert(a == c);
        });
    }
}

void test_im2col() {
    // im2col
    {
        auto input = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 4, 4>);
        std::iota(input.elements().begin(), input.elements().end(), 0.f);
        auto output = ntt::make_tensor<float>(ntt::fixed_shape_v<9, 16>);
        ntt::im2col(input, output, ntt::fixed_shape_v<3, 3>,
                    ntt::fixed_shape_v<1, 1>,
                    ntt::fixed_paddings_v<1, 1, 1, 1>);
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
        auto input = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 4, 4, 4>);
        std::iota(input.elements().begin(), input.elements().end(), 0.f);
        auto packed_input = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 4, 4>);
        ntt::pack(input, packed_input, ntt::fixed_shape_v<1>);
        auto packed_output =
            ntt::make_tensor<ntt::vector<float, 4>>(ntt::fixed_shape_v<9, 16>);
        ntt::im2col(packed_input, packed_output, ntt::fixed_shape_v<3, 3>,
                    ntt::fixed_shape_v<1, 1>, ntt::fixed_paddings_v<1, 1, 1, 1>,
                    ntt::fixed_shape_v<1>, ntt::fixed_shape_v<0>);
        auto unpacked_output =
            ntt::make_tensor<float>(ntt::fixed_shape_v<36, 16>);
        // packed [n,c/4,h,w,4] => [c/4 * h * w, b * oh * ow]
        // so unpack should after reshape
        ntt::unpack(packed_output.reshape(ntt::fixed_shape_v<1, 9, 16>),
                    unpacked_output.reshape(ntt::fixed_shape_v<4, 9, 16>),
                    ntt::fixed_shape_v<0>);
        auto output = ntt::make_tensor<float>(ntt::fixed_shape_v<36, 16>);
        ntt::im2col(input, output, ntt::fixed_shape_v<3, 3>,
                    ntt::fixed_shape_v<1, 1>,
                    ntt::fixed_paddings_v<1, 1, 1, 1>);
        ntt::apply(output.shape(), [&](auto index) {
            NNCASE_UNUSED auto a = output(index);
            NNCASE_UNUSED auto c = unpacked_output(index);
            assert(a == c);
        });
    }
}

void test_concat() {
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 8>);
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 24>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);
    auto pa = ntt::make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<3, 1>);
    auto pb = ntt::make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<3, 2>);
    auto pc = ntt::make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<3, 3>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::pack(tb, pb, ntt::fixed_shape_v<1>);
    ntt::concat(std::make_tuple(pa, pb), pc, 1_dim);
    ntt::unpack(pc, tc, ntt::fixed_shape_v<1>);

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

void test_slice() {
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 24>);
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 8>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 16>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    ntt::slice(ta, tb, ntt::fixed_shape_v<0>, ntt::fixed_shape_v<8>,
               ntt::fixed_shape_v<1>);
    ntt::slice(ta, tc, ntt::fixed_shape_v<8>, fixed_shape_v<24>,
               ntt::fixed_shape_v<1>);
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

void test_transpose() {
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<3, 24>);
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<24, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    ntt::transpose(ta, tb, ntt::fixed_shape_v<1, 0>);
    assert(tb(0, 0) == 0.0f);
    assert(tb(0, 1) == 24.f);
    assert(tb(0, 2) == 48.f);

    auto pa = ntt::make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<3, 3>);
    auto pb = ntt::make_tensor<ntt::vector<float, 8>>(ntt::fixed_shape_v<3, 3>);
    ntt::pack(ta, pa, ntt::fixed_shape_v<1>);
    ntt::transpose(pa, pb.view(), ntt::fixed_shape_v<1, 0>);
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

void test_gather() {
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<6, 3>);
    auto tb = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3>);
    auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 3, 3>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().rbegin(), tb.elements().rend(), 0);
    ntt::gather(ta, tb, tc, 0_dim);
    assert(tc(0, 2, 0) == 0.0f);
    assert(tc(0, 2, 1) == 1.0f);
    assert(tc(0, 2, 2) == 2.0f);
    assert(tc(0, 1, 0) == 3.0f);
    assert(tc(0, 1, 1) == 4.0f);
    assert(tc(0, 1, 2) == 5.0f);
    assert(tc(0, 0, 0) == 6.0f);
    assert(tc(0, 0, 1) == 7.0f);
    assert(tc(0, 0, 2) == 8.0f);

    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 3, 3>);
    auto te = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 2>);
    auto tf = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 1, 2, 3>);
    std::iota(td.elements().begin(), td.elements().end(), 0.f);
    std::iota(te.elements().rbegin(), te.elements().rend(), 0);
    ntt::gather(td, te, tf, 1_dim);
    assert(tf(0, 0, 1, 0) == 0.0f);
    assert(tf(0, 0, 1, 1) == 1.0f);
    assert(tf(0, 0, 1, 2) == 2.0f);
    assert(tf(0, 0, 0, 0) == 3.0f);
    assert(tf(0, 0, 0, 1) == 4.0f);
    assert(tf(0, 0, 0, 2) == 5.0f);
}

void test_pad() {
    auto td = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2, 3>);
    auto te = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 2, 3>);
    std::iota(td.elements().begin(), td.elements().end(), 0.f);
    ntt::pad(td, te, ntt::fixed_paddings_v<0, 7, 0, 0, 0, 0>, 1.3f);
    assert(te(0, 0, 1) == 1.f);
    assert(te(1, 0, 1) == 1.3f);
    assert(te(2, 0, 1) == 1.3f);
    assert(te(3, 0, 1) == 1.3f);
}

void test_reduce() {
    // pack 1d
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 16>);
        auto tav =
            ntt::make_tensor<ntt::vector<float, 4>>(ntt::fixed_shape_v<2, 4>);
        std::fill(ta.elements().begin(), ta.elements().begin() + 16, 1.f);
        std::fill(ta.elements().begin() + 16, ta.elements().end(), 3.2f);
        ntt::pack(ta, tav.view(), ntt::fixed_shape_v<1>);

        auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 1>);
        ntt::reduce_sum(tav, tb, ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1>);
        assert(are_floats_equal(tb(0, 0), 16.f));
        assert(are_floats_equal(tb(1, 0), 51.2f));

        // pack 1d and tiled.
        auto tc = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 1>);
        ntt::reduce_sum(
            tav.view(ntt::make_shape(0, 0), ntt::fixed_shape_v<2, 2>), tc,
            ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1>);
        ntt::reduce_sum<true>(
            tav.view(ntt::make_shape(0, 2), ntt::fixed_shape_v<2, 2>), tc,
            ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1>);
        assert(are_floats_equal(tb(0, 0), 16.f));
        assert(are_floats_equal(tb(1, 0), 51.2f));
    }

    // pack 2d, inner reduce 0
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 32, 8>);
        auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 8>);
        auto upb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 8>);
        auto pa = ntt::make_tensor<ntt::vector<float, 4, 4>>(
            ntt::fixed_shape_v<1, 8, 2>);
        auto pb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 2>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, pa.view(), ntt::fixed_shape_v<1, 2>);

        ntt::reduce_sum(ta, tb, ntt::fixed_shape_v<1>);

        ntt::reduce_sum(pa, pb, ntt::fixed_shape_v<1>,
                        ntt::fixed_shape_v<1, 2>);

        ntt::unpack(pb, upb.view(), ntt::fixed_shape_v<2>);
        ntt::apply(tb.shape(), [&]([[maybe_unused]] auto index) {
            assert(tb(index) == upb(index));
        });

        // tiling on reduced axis
        auto pc = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 1, 2>);
        auto upc = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 8>);
        ntt::reduce_sum(
            pa.view(ntt::make_shape(0, 0, 0), ntt::fixed_shape_v<1, 4, 2>), pc,
            ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1, 2>);
        ntt::reduce_sum<true>(
            pa.view(ntt::make_shape(0, 4, 0), ntt::fixed_shape_v<1, 4, 2>), pc,
            ntt::fixed_shape_v<1>, ntt::fixed_shape_v<1, 2>);

        ntt::unpack(pc, upc.view(), ntt::fixed_shape_v<2>);
        ntt::apply(tb.shape(), [&]([[maybe_unused]] auto index) {
            assert(tb(index) == upc(index));
        });
    }

    // pack 2d, inner reduce 1
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 8, 16>);
        auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 8, 1>);
        auto upb = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 8, 1>);
        auto upc = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 8, 1>);
        auto pa = ntt::make_tensor<ntt::vector<float, 4, 4>>(
            ntt::fixed_shape_v<1, 2, 4>);
        auto pb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 2, 1>);
        auto pc = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 2, 1>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, pa.view(), ntt::fixed_shape_v<1, 2>);

        ntt::reduce_mean(ta, tb, ntt::fixed_shape_v<2>);
        ntt::reduce_mean(pa, pb, ntt::fixed_shape_v<2>,
                         ntt::fixed_shape_v<1, 2>);

        ntt::unpack(pb, upb.view(), ntt::fixed_shape_v<1>);
        ntt::apply(tb.shape(), [&]([[maybe_unused]] auto index) {
            assert(tb(index) == upb(index));
        });

        // tiling on reduced axis
        ntt::reduce_max(ta, tb, ntt::fixed_shape_v<2>);
        ntt::reduce_max(
            pa.view(ntt::make_shape(0, 0, 0), ntt::fixed_shape_v<1, 2, 1>), pc,
            ntt::fixed_shape_v<2>, ntt::fixed_shape_v<1, 2>);
        ntt::reduce_max<true>(
            pa.view(ntt::make_shape(0, 0, 1), ntt::fixed_shape_v<1, 2, 2>), pc,
            ntt::fixed_shape_v<2>, ntt::fixed_shape_v<1, 2>);
        ntt::reduce_max<true>(
            pa.view(ntt::make_shape(0, 0, 3), ntt::fixed_shape_v<1, 2, 1>), pc,
            ntt::fixed_shape_v<2>, ntt::fixed_shape_v<1, 2>);

        ntt::unpack(pc, upc.view(), ntt::fixed_shape_v<1>);
        ntt::apply(tb.shape(), [&]([[maybe_unused]] auto index) {
            assert(tb(index) == upc(index));
        });
    }
}

void test_cast() {
    // normal cast
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 16>);
        auto tb = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<1, 16>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::cast(ta, tb.view());
        assert(tb(0, 0) == 0);
        assert(tb(0, 2) == 2);
    }

    // packed cast
    {
        auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 64, 32>);
        auto tb = ntt::make_tensor<ntt::vector<float, 4>>(
            ntt::fixed_shape_v<1, 16, 32>);
        std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
        ntt::pack(ta, tb.view(), ntt::fixed_shape_v<1>);
        auto tc = ntt::make_tensor<ntt::vector<int32_t, 4>>(
            ntt::fixed_shape_v<1, 16, 32>);
        ntt::cast(tb, tc);
        assert(tc(0, 0, 0)(0) == 0);
        assert(tc(0, 0, 0)(1) == 32);
        assert(tc(0, 0, 0)(2) == 64);
    }
}

void test_expand() {
    // [1, 3, 5, 7] strides = [3*5*7, 5*7, 7 , 1]
    //          [1] strides = [1] -> shape [1, 3, 5, 7] strides = [0, 0, 0, 1]
    // [1, 3, 5, 7] strides = [3*5*7, 5*7, 7 , 1]
    //          [7] strides = [1] -> shape [1, 3, 5, 7] strides = [0, 0, 0, 1]
    // [1, 3, 5, 7] strides = [3*5*7, 5*7, 7 , 1]
    //       [5, 7] strides = [7, 1] -> shape [1, 3, 5, 7] strides = [0, 0, 7,
    //       1]
    // [1, 3, 5, 7] strides = [3*5*7, 5*7, 7 , 1]
    //       [5, 1] strides = [1, 1] -> shape [1, 3, 5, 7] strides = [0, 0, 1,
    //       1]
    auto ta = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 2>);
    auto tb = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 2>);
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    ntt::expand(ta, tb.view());
    assert(are_floats_equal(tb(0, 0), 0.f));
    assert(are_floats_equal(tb(0, 1), 1.f));
    assert(are_floats_equal(tb(1, 0), 0.f));
    assert(are_floats_equal(tb(1, 1), 1.f));
}

void test_where() {
    auto tcond = ntt::make_tensor<bool>(ntt::fixed_shape_v<2, 2>);
    auto tx = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 2>);
    auto ty = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 2>);
    auto tout = ntt::make_tensor<float>(ntt::fixed_shape_v<2, 2>);
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

#if 0
void test_reduce_arg() {
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
#endif

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

    test_matmul_normal();
    test_pack();
    test_im2col();
    test_concat();
    test_slice();
    test_transpose();
    test_gather();
    test_pad();
    test_reduce();
    test_cast();
    test_expand();
    test_where();
#if 0
    test_matmul_transpose_b();
    test_caching();
    test_unary_binary();
    test_tensor_view();
    test_reduce_arg();
#endif

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
