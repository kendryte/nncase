#pragma once

#include <array>
#include <cstddef>
#include <gsl/gsl-lite.hpp>
#include <iostream>
#include <nncase/runtime/small_vector.hpp>
#include <numeric>
#include <vector>

using dims_t = itlib::small_vector<size_t, 8>;
using strides_t = itlib::small_vector<size_t, 8>;

void print_vec(itlib::small_vector<size_t, 8> vec) {
    for (const size_t v : vec) {
        std::cout << std::to_string(v) << ", ";
    }
    std::cout << std::endl;
}

template <class TShape> inline size_t compute_size(const TShape &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
}

template <class TShape>
inline size_t compute_size(const TShape &shape, const TShape &strides) {
    size_t max_stride = 1, max_shape = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if ((shape[i] == 1 ? 0 : strides[i]) >= max_stride) {
            max_stride = strides[i];
            max_shape = shape[i];
        }
    }
    size_t size = max_stride * max_shape;
    return size;
}

template <class shape_type, class strides_type>
inline std::size_t compute_strides(const shape_type &shape,
                                   strides_type &strides) {
    using strides_value_type = typename std::decay_t<strides_type>::value_type;
    strides_value_type data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i) {
        strides[i - 1] = data_size;
        data_size =
            strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
    }
    return static_cast<std::size_t>(data_size);
}

inline strides_t get_default_strides(dims_t shape) {
    strides_t strides(shape.size());
    compute_strides(shape, strides);
    return strides;
}

template <class offset_type, class S, class It>
inline offset_type element_offset(const S &strides, It first,
                                  It last) noexcept {
    using difference_type = typename std::iterator_traits<It>::difference_type;
    auto size = static_cast<difference_type>((std::min)(
        static_cast<size_t>(std::distance(first, last)), strides.size()));
    return std::inner_product(last - size, last, strides.cend() - size,
                              offset_type(0));
}

inline size_t offset(gsl::span<const size_t> strides,
                     gsl::span<const size_t> index) {
    // scalar
    if (strides.size() == 0 || index.size() == 0) {
        return 0;
    }
    assert(strides.size() == index.size());
    return element_offset<size_t>(strides, index.begin(), index.end());
}

inline bool is_shape_equal(const dims_t &a, const dims_t &b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}
