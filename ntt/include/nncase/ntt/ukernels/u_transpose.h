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
#include <type_traits>

namespace nncase::ntt {
namespace ukernels {

struct segment {
    size_t start;
    size_t length;
    size_t index;
};

template <IsFixedDims TPerm, size_t Rank>
constexpr std::array<segment, Rank> get_segments() {

    std::array<segment, Rank> segments;

    size_t segment_count = 0;
    size_t start = 0;
    size_t length = 1;

    for (size_t i = 1; i < TPerm::rank(); ++i) {
        if (TPerm::at(i) == TPerm::at(i - 1) + 1) {
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
                if (TPerm::at(segments[i].start) >
                    TPerm::at(segments[j].start)) {
                    ++cnt;
                }
            }
        }
        segments[i].index = cnt;
    }

    return segments;
}

template <IsFixedDims TPerm, IsFixedTensor TIn, size_t Rank>
constexpr std::array<size_t, Rank> compress_dimensions() {

    constexpr auto shape = TIn::shape();
    constexpr auto rank = TIn::rank();

    constexpr std::array<segment, Rank> segments = get_segments<TPerm, Rank>();

    std::array<size_t, rank> new_shape{};
    std::array<size_t, Rank> dims_compressed{};

    for (size_t i = 0; i < Rank; i += 1) {
        size_t compressed_dim = shape[TPerm::at(segments[i].start)];
        for (size_t j = 1; j < segments[i].length; j++) {
            compressed_dim *= shape[TPerm::at(segments[i].start + j)];
        }
        new_shape[TPerm::at(segments[i].start)] = compressed_dim;
    }

    size_t cnt = 0;
    for (size_t i = 0; i < rank; ++i) {
        if (new_shape[i] != 0) {
            dims_compressed[cnt++] = new_shape[i];
        }
    }

    return dims_compressed;
}

template <IsFixedDims TPerm, size_t Rank>
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
    // std::array<size_t, Rank> stride_compressed{};

    for (size_t i = 0; i < Rank; ++i) {
        dims_transed[i] = dims[perm[i]];
    }

    return dims_transed;
}

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, size_t Rank,
          bool Arch>
class u_transpose;

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, bool Arch>
class u_transpose<TPerm, TIn, TOut, 1, Arch> {
  public:
    constexpr void operator()(const TIn &input, TOut &output) noexcept {

        auto domain = input.shape();
        auto out_index = ranked_shape<domain.rank()>{};
        apply(domain, [&](auto index) {
            loop<domain.rank()>(
                [&](auto i) { out_index[i] = index[TPerm::at(i)]; });
            output(out_index) = input(index);
        });
    }
};

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, bool Arch>
class u_transpose<TPerm, TIn, TOut, 2, Arch> {
  public:
    __attribute__((always_inline)) constexpr void
    operator()(const TIn &input, TOut &output) noexcept {

        constexpr std::array<size_t, 2> dims_compressed =
            compress_dimensions<TPerm, TIn, 2>();
        constexpr std::array<size_t, 2> perm_compressed =
            compress_perm<TPerm, 2>();
        constexpr auto shape_transed =
            trans_shape<2>(dims_compressed, perm_compressed);
        constexpr auto data_length = TIn::size();

        using TElem = typename TIn::element_type;

        auto compressed_input = ntt::tensor_view<
            TElem, ntt::fixed_shape<dims_compressed[0], dims_compressed[1]>>(
            std::span<TElem, data_length>((TElem *)(input.elements().data()),
                                          data_length));

        // auto t1 = get_cpu_cycle();
        auto compressed_output = ntt::tensor_view<
            TElem, ntt::fixed_shape<shape_transed[0], shape_transed[1]>>(
            std::span<TElem, data_length>((TElem *)(output.elements().data()),
                                          data_length));

        constexpr auto domain = compressed_input.shape();
        auto out_index = ranked_shape<2>{};
        apply(domain, [&](auto index) {
            loop<2>([&](auto i) { out_index[i] = index[perm_compressed[i]]; });
            compressed_output(out_index) = compressed_input(index);
        });
    }
};

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, bool Arch>
class u_transpose<TPerm, TIn, TOut, 3, Arch> {
  public:
    __attribute__((always_inline)) constexpr void
    operator()(const TIn &input, TOut &output) noexcept {

        constexpr std::array<size_t, 3> dims_compressed =
            compress_dimensions<TPerm, TIn, 3>();
        constexpr std::array<size_t, 3> perm_compressed =
            compress_perm<TPerm, 3>();
        constexpr auto shape_transed =
            trans_shape<3>(dims_compressed, perm_compressed);
        constexpr auto data_length = TIn::size();

        using TElem = typename TIn::element_type;

        auto compressed_input = ntt::tensor_view<
            TElem, ntt::fixed_shape<dims_compressed[0], dims_compressed[1],
                                    dims_compressed[2]>>(
            std::span<TElem, data_length>((TElem *)(input.elements().data()),
                                          data_length));

        // auto t1 = get_cpu_cycle();
        auto compressed_output = ntt::tensor_view<
            TElem, ntt::fixed_shape<shape_transed[0], shape_transed[1],
                                    shape_transed[2]>>(
            std::span<TElem, data_length>((TElem *)(output.elements().data()),
                                          data_length));

        constexpr auto domain = compressed_input.shape();
        auto out_index = ranked_shape<3>{};
        apply(domain, [&](auto index) {
            loop<3>([&](auto i) { out_index[i] = index[perm_compressed[i]]; });
            compressed_output(out_index) = compressed_input(index);
        });
    }
};

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, bool Arch>
class u_transpose<TPerm, TIn, TOut, 4, Arch> {
  public:
    __attribute__((always_inline)) constexpr void
    operator()(const TIn &input, TOut &output) noexcept {
        constexpr std::array<size_t, 4> dims_compressed =
            compress_dimensions<TPerm, TIn, 4>();
        constexpr std::array<size_t, 4> perm_compressed =
            compress_perm<TPerm, 4>();
        constexpr auto shape_transed =
            trans_shape<4>(dims_compressed, perm_compressed);
        constexpr auto data_length = TIn::size();

        using TElem = typename TIn::element_type;

        auto compressed_input = ntt::tensor_view<
            TElem, ntt::fixed_shape<dims_compressed[0], dims_compressed[1],
                                    dims_compressed[2], dims_compressed[3]>>(
            std::span<TElem, data_length>((TElem *)(input.elements().data()),
                                          data_length));

        auto compressed_output = ntt::tensor_view<
            TElem, ntt::fixed_shape<shape_transed[0], shape_transed[1],
                                    shape_transed[2], shape_transed[3]>>(
            std::span<TElem, data_length>((TElem *)(output.elements().data()),
                                          data_length));

        constexpr auto domain = compressed_input.shape();
        auto out_index = ranked_shape<4>{};
        apply(domain, [&](auto index) {
            loop<4>([&](auto i) { out_index[i] = index[perm_compressed[i]]; });
            compressed_output(out_index) = compressed_input(index);
        });
    }
};
} // namespace ukernels

template <IsFixedDims TPerm, IsFixedTensor TIn, IsFixedTensor TOut, size_t Rank>
constexpr void u_transpose(const TIn &input, TOut &&output) noexcept {
    ukernels::u_transpose<TPerm, std::decay_t<TIn>, std::decay_t<TOut>, Rank,
                          true>
        impl;

    impl(input, output);
}

} // namespace nncase::ntt
