#pragma once
#include "runtime_utils.h"
#include <cmath>
#include <unordered_map>

enum class reduce_arg_op_t : uint8_t {
    arg_min = 0,
    arg_max = 1,
};

namespace kernels {
template <class TReducer, class TOutput, class T>
void reduce_arg_impl(TReducer &&reducer, T init_value, const T *input,
                     TOutput *output, gsl::span<const size_t> in_shape,
                     gsl::span<const size_t> out_shape,
                     gsl::span<const size_t> in_strides,
                     gsl::span<const size_t> out_strides,
                     gsl::span<const size_t> axes, bool keep_dims,
                     bool select_last_idx) noexcept {
    const float epsilon = 0.000001f;

    // init with init_value
    auto output_size = compute_size(out_shape);
    auto ptr = (T *)runtime_util->malloc(output_size * sizeof(T));
    apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        ptr[offset(out_strides, index)] = init_value;
    });

    // collect all min/max indices
    auto out_map = (TOutput *)runtime_util->malloc(output_size * 2 * sizeof(TOutput));
    // std::unordered_map<size_t, std::vector<TOutput>> out_map;
    apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto src = input[offset(in_strides, index)];
        auto out_idx =
            offset(out_strides, get_reduced_offset(index, axes, keep_dims));
        auto &dst = ptr[out_idx];
        auto ret = reducer(src, dst);
        if (ret) {
            out_map[out_idx * 2 + 0] = index[axes[0]];
            out_map[out_idx * 2 + 1] = index[axes[0]];
            dst = src;
        } else if (std::fabs(src - dst) < epsilon) {
            out_map[out_idx * 2 + 1] = index[axes[0]];
        }
    });

    // update min/max idx
    apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        auto out_idx = offset(out_strides, index);
        output[out_idx] = select_last_idx ? out_map[out_idx * 2 + 0]
                                          : out_map[out_idx * 2 + 1];
    });

    runtime_util->free(out_map);
    runtime_util->free(ptr);
}

template <class Tin, class Tout>
void reduce_arg(reduce_arg_op_t op, const Tin *input, Tout *output,
                gsl::span<const size_t> in_shape,
                gsl::span<const size_t> in_strides,
                gsl::span<const size_t> out_strides,
                gsl::span<const size_t> axes, bool keep_dims,
                bool select_last_idx) noexcept {
    auto out_shape = get_reduced_shape(in_shape, axes, keep_dims);
    switch (op) {
    case reduce_arg_op_t::arg_min:
        return reduce_arg_impl([](Tin a, Tin b) { return a < b; },
                               std::numeric_limits<Tin>::max(), input, output,
                               in_shape, out_shape, in_strides, out_strides,
                               axes, keep_dims, select_last_idx);
    case reduce_arg_op_t::arg_max:
        return reduce_arg_impl([](Tin a, Tin b) { return a > b; },
                               std::numeric_limits<Tin>::lowest(), input,
                               output, in_shape, out_shape, in_strides,
                               out_strides, axes, keep_dims, select_last_idx);
    }
}

} // namespace kernels