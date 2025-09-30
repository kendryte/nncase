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
#pragma once
#include "../loop.h"
#include <cstddef>
#include <iostream>
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {
template <Tensor TIn, class TOut, FixedDimensions TPerms, bool Arch>
class u_transpose_impl {

  public:
    constexpr void operator()(const TIn &input, TOut &output, const TPerms &) {
        constexpr auto rank = TIn::rank();
        constexpr TPerms perm_const;
        constexpr auto pos_perms = positive_axes(perm_const, rank);
        // printf("%s, %d: \n", __FILE__, __LINE__);
        ntt::apply(input.shape(), [&](auto index) {
            auto out_index = generate_shape<rank>(
                [&](auto i) { return index[pos_perms[i]]; });
            output(out_index) = input(index);
        });
    }
};

#define DEFINE_U_TRANSPOSE_IMPL_4D(PERM0, PERM1, PERM2, PERM3, ACCESS_EXPR)    \
    template <Tensor TIn, class TOut, bool Arch>                               \
    class u_transpose_impl<TIn, TOut,                                          \
                           fixed_shape_t<PERM0, PERM1, PERM2, PERM3>, Arch> {  \
      public:                                                                  \
        constexpr void                                                         \
        operator()(const TIn &input, TOut &output,                             \
                   const fixed_shape_t<PERM0, PERM1, PERM2, PERM3> &) {        \
            for (auto i = 0; i < input.shape()[0]; ++i)                        \
                for (auto j = 0; j < input.shape()[1]; ++j)                    \
                    for (auto k = 0; k < input.shape()[2]; ++k)                \
                        for (auto l = 0; l < input.shape()[3]; ++l)            \
                            output ACCESS_EXPR = input(i, j, k, l);            \
        }                                                                      \
    }

#define DEFINE_U_TRANSPOSE_IMPL_3D(PERM0, PERM1, PERM2, ACCESS_EXPR)           \
    template <Tensor TIn, class TOut, bool Arch>                               \
    class u_transpose_impl<TIn, TOut, fixed_shape_t<PERM0, PERM1, PERM2>,      \
                           Arch> {                                             \
      public:                                                                  \
        constexpr void                                                         \
        operator()(const TIn &input, TOut &output,                             \
                   const fixed_shape_t<PERM0, PERM1, PERM2> &) {               \
            for (auto i = 0; i < input.shape()[0]; ++i)                        \
                for (auto j = 0; j < input.shape()[1]; ++j)                    \
                    for (auto k = 0; k < input.shape()[2]; ++k)                \
                        output ACCESS_EXPR = input(i, j, k);                   \
        }                                                                      \
    }

#define DEFINE_U_TRANSPOSE_IMPL_2D(PERM0, PERM1, ACCESS_EXPR)                  \
    template <Tensor TIn, class TOut, bool Arch>                               \
    class u_transpose_impl<TIn, TOut, fixed_shape_t<PERM0, PERM1>, Arch> {     \
      public:                                                                  \
        constexpr void operator()(const TIn &input, TOut &output,              \
                                  const fixed_shape_t<PERM0, PERM1> &) {       \
            for (auto i = 0; i < input.shape()[0]; ++i)                        \
                for (auto j = 0; j < input.shape()[1]; ++j)                    \
                    output ACCESS_EXPR = input(i, j);                          \
        }                                                                      \
    }

DEFINE_U_TRANSPOSE_IMPL_4D(0, 1, 2, 3, (i, j, k, l));
DEFINE_U_TRANSPOSE_IMPL_4D(0, 1, 3, 2, (i, j, l, k));
DEFINE_U_TRANSPOSE_IMPL_4D(0, 2, 1, 3, (i, k, j, l));
DEFINE_U_TRANSPOSE_IMPL_4D(0, 2, 3, 1, (i, k, l, j));
DEFINE_U_TRANSPOSE_IMPL_4D(0, 3, 1, 2, (i, l, j, k));
DEFINE_U_TRANSPOSE_IMPL_4D(0, 3, 2, 1, (i, l, k, j));

DEFINE_U_TRANSPOSE_IMPL_4D(1, 0, 2, 3, (j, i, k, l));
DEFINE_U_TRANSPOSE_IMPL_4D(1, 0, 3, 2, (j, i, l, k));
DEFINE_U_TRANSPOSE_IMPL_4D(1, 2, 0, 3, (j, k, i, l));
DEFINE_U_TRANSPOSE_IMPL_4D(1, 2, 3, 0, (j, k, l, i));
DEFINE_U_TRANSPOSE_IMPL_4D(1, 3, 0, 2, (j, l, i, k));
DEFINE_U_TRANSPOSE_IMPL_4D(1, 3, 2, 0, (j, l, k, i));

DEFINE_U_TRANSPOSE_IMPL_4D(2, 0, 1, 3, (k, i, j, l));
DEFINE_U_TRANSPOSE_IMPL_4D(2, 0, 3, 1, (k, i, l, j));
DEFINE_U_TRANSPOSE_IMPL_4D(2, 1, 0, 3, (k, j, i, l));
DEFINE_U_TRANSPOSE_IMPL_4D(2, 1, 3, 0, (k, j, l, i));
DEFINE_U_TRANSPOSE_IMPL_4D(2, 3, 0, 1, (k, l, i, j));
DEFINE_U_TRANSPOSE_IMPL_4D(2, 3, 1, 0, (k, l, j, i));

DEFINE_U_TRANSPOSE_IMPL_4D(3, 0, 1, 2, (l, i, j, k));
DEFINE_U_TRANSPOSE_IMPL_4D(3, 0, 2, 1, (l, i, k, j));
DEFINE_U_TRANSPOSE_IMPL_4D(3, 1, 0, 2, (l, j, i, k));
DEFINE_U_TRANSPOSE_IMPL_4D(3, 1, 2, 0, (l, j, k, i));
DEFINE_U_TRANSPOSE_IMPL_4D(3, 2, 0, 1, (l, k, i, j));
DEFINE_U_TRANSPOSE_IMPL_4D(3, 2, 1, 0, (l, k, j, i));

// 3D
DEFINE_U_TRANSPOSE_IMPL_3D(0, 1, 2, (i, j, k));
DEFINE_U_TRANSPOSE_IMPL_3D(0, 2, 1, (i, k, j));
DEFINE_U_TRANSPOSE_IMPL_3D(1, 0, 2, (j, i, k));
DEFINE_U_TRANSPOSE_IMPL_3D(1, 2, 0, (j, k, i));
DEFINE_U_TRANSPOSE_IMPL_3D(2, 0, 1, (k, i, j));
DEFINE_U_TRANSPOSE_IMPL_3D(2, 1, 0, (k, j, i));

// 2D
DEFINE_U_TRANSPOSE_IMPL_2D(0, 1, (i, j));
DEFINE_U_TRANSPOSE_IMPL_2D(1, 0, (j, i));
}; // namespace ukernels
namespace u_transpose_detail {
struct segment {
    size_t start;
    size_t length;
    size_t index;
};

template <FixedDimensions TPerms, size_t Rank>
constexpr std::array<segment, Rank> get_segments() {

    constexpr TPerms perms;

    std::array<segment, Rank> segments;

    size_t segment_count = 0;
    size_t start = 0;
    size_t length = 1;

    for (size_t i = 1; i < TPerms::rank(); ++i) {
        if (perms[i] == perms[i - 1] + 1) {
            ++length;
        } else {
            segments[segment_count].start = start;
            segments[segment_count].length = length;
            ++segment_count;
            start = i;
            length = 1;
        }
    }

    segments[segment_count].start = start;
    segments[segment_count].length = length;
    ++segment_count;

    for (size_t i = 0; i < Rank; ++i) {
        size_t cnt = 0;
        for (size_t j = 0; j < Rank; ++j) {
            if (i != j) {
                if (perms[segments[i].start] > perms[segments[j].start]) {
                    ++cnt;
                }
            }
        }
        segments[i].index = cnt;
    }

    return segments;
}

template <FixedDimensions TPerms, Tensor TIn, size_t Rank>
constexpr std::array<size_t, Rank> compress_dimensions(const TIn &input) {

    constexpr TPerms perms;
    auto shape = input.shape();
    constexpr auto rank = TIn::rank();

    constexpr std::array<segment, Rank> segments = get_segments<TPerms, Rank>();

    std::array<size_t, rank> new_shape{};
    std::array<size_t, Rank> dims_compressed{};

    for (size_t i = 0; i < Rank; i += 1) {
        size_t compressed_dim = shape[perms[segments[i].start]];
        for (size_t j = 1; j < segments[i].length; j++) {
            compressed_dim *= shape[perms[segments[i].start + j]];
        }
        new_shape[perms[segments[i].start]] = compressed_dim;
    }

    size_t cnt = 0;
    for (size_t i = 0; i < rank; ++i) {
        if (new_shape[i] != 0) {
            dims_compressed[cnt++] = new_shape[i];
        }
    }

    return dims_compressed;
}

template <FixedDimensions TPerm, size_t Rank>
constexpr std::array<size_t, Rank> compress_perm() {
    std::array<size_t, Rank> new_dims{};

    constexpr std::array<segment, Rank> segments = get_segments<TPerm, Rank>();

    for (size_t i = 0; i < Rank; i += 1) {
        new_dims[i] = segments[i].index;
    }

    return new_dims;
}

template <size_t Rank>
constexpr std::array<size_t, Rank>
trans_shape(const std::array<size_t, Rank> dims,
            const std::array<size_t, Rank> perm) {
    std::array<size_t, Rank> dims_transed{};

    for (size_t i = 0; i < Rank; ++i) {
        dims_transed[i] = dims[perm[i]];
    }

    return dims_transed;
}

template <Tensor TIn, FixedDimensions TPerms>
bool is_segment_contiguous(const TIn &input, const segment &seg) {
    constexpr TPerms perms;
    auto shape = input.shape();
    auto strides = input.strides();
    
    // 单个维度总是连续的
    // if (seg.length <= 1) 
    // {
    //     printf("segment length <= 1, always contiguous\n");
    //     return true;
    // }
    
    // 检查segment内相邻维度是否满足连续性条件
    for (size_t i = 0; i< seg.length; ++i) {
        size_t axis_cur = perms[seg.start + i];
        size_t axis_next = perms[seg.start + i + 1];
        
        // 连续性条件：stride[axis_cur] == shape[axis_next] * stride[axis_next]
        // printf("strides[axis_cur] != shape[axis_next] * strides[axis_next]: %zu != %zu * %zu\n", strides[axis_cur], shape[axis_next], strides[axis_next]);
        if (strides[axis_cur] != shape[axis_next] * strides[axis_next]) {
            return false;
        }
    }
    return true;
}

// 修改后的compress_dimensions，考虑stride连续性
template <FixedDimensions TPerms, Tensor TIn, size_t MaxRank>
auto compress_dimensions_with_strides(const TIn &input) {
    constexpr TPerms perms;
    auto shape = input.shape();
    auto strides = input.strides();
    constexpr auto rank = TIn::rank();

    constexpr std::array<segment, MaxRank> segments = get_segments<TPerms, MaxRank>();
    
    // 检查每个segment的连续性，决定实际的segment数量
    size_t actual_segments = 0;
    std::array<segment, MaxRank> valid_segments{};
    std::array<size_t, MaxRank> compressed_dims{};
    std::array<size_t, MaxRank> compressed_strides{};
    
    for (size_t i = 0; i < MaxRank && segments[i].length > 0; ++i) {
        const auto &seg = segments[i];
        
        if (is_segment_contiguous<TIn, TPerms>(input, seg)) {
            // segment连续，可以压缩
            size_t compressed_dim = shape[perms[seg.start]];
            size_t compressed_stride = strides[perms[seg.start + seg.length - 1]];
            
            for (size_t j = 1; j < seg.length; ++j) {
                compressed_dim *= shape[perms[seg.start + j]];
            }
            
            valid_segments[actual_segments] = {actual_segments, 1, seg.index};
            compressed_dims[actual_segments] = compressed_dim;
            compressed_strides[actual_segments] = compressed_stride;
            actual_segments++;
        } else {
            // segment不连续，拆分为单独的维度
            for (size_t j = 0; j < seg.length; ++j) {
                size_t axis = perms[seg.start + j];
                valid_segments[actual_segments] = {actual_segments, 1, actual_segments};
                compressed_dims[actual_segments] = shape[axis];
                compressed_strides[actual_segments] = strides[axis];
                actual_segments++;
            }
        }
    }
    
    struct CompressResult {
        size_t segment_count;
        std::array<size_t, MaxRank> dims;
        std::array<size_t, MaxRank> strides;
        std::array<segment, MaxRank> segments;
    };
    
    return CompressResult{actual_segments, compressed_dims, compressed_strides, valid_segments};
}

} // namespace u_transpose_detail

template <Tensor TIn, class TOut, FixedDimensions TPerms, size_t Segments,
          size_t... Index>
    requires(bool(TIn::rank() == std::decay_t<TOut>::rank()) &&
             bool(TIn::rank() == TPerms::rank()))
void u_transpose(const TIn &input, TOut &output, const TPerms &perms,
                 std::index_sequence<Index...>) {
    // 检查是否所有segment都连续，决定是否可以使用压缩优化
    constexpr std::array<u_transpose_detail::segment, Segments> segments =
        u_transpose_detail::get_segments<TPerms, Segments>();
    
    bool can_compress = true;
    for (size_t i = 0; i < Segments && can_compress; ++i) {
        if (!u_transpose_detail::is_segment_contiguous<TIn, TPerms>(input, segments[i])) {
            can_compress = false;
        }
    }
    
    if (!can_compress) {
        // 无法压缩，使用通用实现
        // printf("%s, %d\n <stride: in[%zu, %zu], out[%zu, %zu]>\n", __FILE__, __LINE__, input.strides()[0], input.strides()[1], output.strides()[0], output.strides()[1]);
        ukernels::u_transpose_impl<TIn, std::decay_t<TOut>, TPerms, true> impl;
        impl(input, output, perms);
        return;
    }
    // printf("%s, %d\n <stride: in[%zu, %zu], out[%zu, %zu]>\n", __FILE__, __LINE__, input.strides()[0], input.strides()[1],
    //                    output.strides()[0], output.strides()[1]);
    const std::array<size_t, Segments> dims_compressed =
        u_transpose_detail::compress_dimensions<TPerms, TIn, Segments>(input);
    constexpr std::array<size_t, Segments> perm_compressed =
        u_transpose_detail::compress_perm<TPerms, Segments>();
    auto shape_transed = u_transpose_detail::trans_shape<Segments>(
        dims_compressed, perm_compressed);
        
    using TElem = typename TIn::element_type;

    auto compressed_input = make_tensor_view(
        input.elements(), make_shape(dims_compressed[Index]...));

    auto compressed_output = make_tensor_view(
        output.elements(), make_shape(shape_transed[Index]...));
    // printf("%s, %d\n <stride: in[%zu, %zu], out[%zu, %zu]>\n", __FILE__, __LINE__, compressed_input.strides()[0], compressed_input.strides()[1],
    //        compressed_output.strides()[0], compressed_output.strides()[1]);

    using TInCompressed = decltype(compressed_input);
    using TOutCompressed = decltype(compressed_output);
    using TPermsCompressed = fixed_shape_t<perm_compressed[Index]...>;
    ukernels::u_transpose_impl<TInCompressed, std::decay_t<TOutCompressed>,
                               TPermsCompressed, true>
        impl;
    impl(compressed_input, compressed_output,
         fixed_shape_v<perm_compressed[Index]...>);
}

} // namespace nncase::ntt
