#pragma once

#include <cstddef>
#include <gsl/gsl-lite.hpp>
#include <runtime_types.h>
#include "../../method_table_def.h"

using namespace nncase::runtime::cpu;

static nncase_mt_t nncase_mt;
static runtime_util_mt runtime_util;

void print_vec(itlib::small_vector<size_t, 8> vec) {
    for (const size_t v : vec) {
        runtime_util.printf("%zu, ", v);
    }
    runtime_util.printf("\n");
}

template <class TShape> inline size_t compute_size(const TShape &shape) {
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    return size;
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
inline size_t compute_strides(const shape_type &shape, strides_type &strides) {
    size_t data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i) {
        strides[i - 1] = data_size;
        data_size = strides[i - 1] * static_cast<size_t>(shape[i - 1]);
    }
    return static_cast<size_t>(data_size);
}

inline strides_t get_default_strides(dims_t shape) {
    strides_t strides(shape.size());
    compute_strides(shape, strides);
    return strides;
}

template <class offset_type, class S>
inline offset_type element_offset(const S &strides, const S &index) noexcept {
    offset_type size = 0;
    for (auto i = 0; i < strides.size(); i++) {
        size += strides[i] * index[i];
    }
    return size;
}

inline size_t offset(gsl::span<const size_t> strides,
                     gsl::span<const size_t> index) {
    // scalar
    if (strides.size() == 0 || index.size() == 0) {
        return 0;
    }
    runtime_util.rt_assert(strides.size() == index.size(),
                           (char *)"strides and index must have the same rank");
    return element_offset<size_t>(strides, index);
}

inline bool is_shape_equal(const dims_t &a, const dims_t &b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

inline bool is_scalar(gsl::span<const size_t> t) noexcept { return t.empty(); }

inline dims_t to_nd(dims_t in_shape, size_t n) {
    auto size = n - in_shape.size();
    for (size_t i = 0; i < size; ++i) {
        in_shape.insert(in_shape.begin(), 1);
    }
    return in_shape;
}

inline std::tuple<dims_t, strides_t> to_nd(dims_t in_shape, strides_t in_stride,
                                           size_t n) {
    auto size = n - in_shape.size();
    auto stride = *in_stride.begin();
    for (size_t i = 0; i < size; ++i) {
        in_shape.insert(in_shape.begin(), 1);
        in_stride.insert(in_stride.begin(), stride);
    }
    return std::make_tuple(in_shape, in_stride);
}

template <class T> inline bool is_contiguous(const T &shape, const T &strides) {
    size_t data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i) {
        if (strides[i - 1] != data_size) {
            return false;
        }
        data_size *= shape[i - 1];
    }
    return true;
}

inline int
get_last_not_contiguous_index(gsl::span<const size_t> strides,
                              gsl::span<const size_t> default_strides) {
    for (int i = strides.size() - 1; i >= 0; --i) {
        if (strides[i] != default_strides[i]) {
            return i + 1;
        }
    }
    return -1;
}

// template <typename T>
// inline void span_copy(gsl::span<T> dest, gsl::span<T> src) {
//     std::copy(src.data(), src.data()+src.size(), dest.data());
// }

// template <typename T>
// inline void span_equal(gsl::span<T> dest, gsl::span<T> src) {
//     std::copy(src.begin(), src.end(), dest.begin());
// }

template <typename T> double dot(const T *v1, const T *v2, size_t size) {
    double ret = 0.f;
    for (size_t i = 0; i < size; i++) {
        ret += v1[i] * v2[i];
    }

    return ret;
}

template <typename T> double cosine(const T *v1, const T *v2, size_t size) {
    for (size_t i = 0; i < 10; i++) {
        runtime_util.printf("%f, %f\n", (float)v1[i], (float)v2[i]);
        ;
    }
    return dot(v1, v2, size) / ((nncase_mt.float_unary_sqrt(dot(v1, v1, size)) *
                                 nncase_mt.float_unary_sqrt(dot(v2, v2, size))));
}

inline dims_t get_reduced_offset(gsl::span<const size_t> in_offset,
                                 gsl::span<const size_t> axis, bool keep_dims) {
    if (in_offset.size() == 0) {
        return in_offset;
    }
    dims_t off;
    off.reserve(in_offset.size() - (keep_dims ? 0 : axis.size()));
    for (size_t i = 0; i < in_offset.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < axis.size(); j++) {
            if (i == axis[j]) {
                found = true;
                break;
            }
        }
        if (!found) {
            off.push_back(in_offset[i]);
        } else {
            if (keep_dims)
                off.push_back(0);
        }
    }

    return off;
}

inline dims_t get_reduced_offset(gsl::span<const size_t> in_offset,
                                 gsl::span<const size_t> reduced_shape) {
    dims_t off(reduced_shape.size());
    const auto dims_ext = in_offset.size() - reduced_shape.size();
    for (size_t i = 0; i < reduced_shape.size(); i++) {
        if (in_offset[i + dims_ext] >= reduced_shape[i])
            off[i] = 0;
        else
            off[i] = in_offset[i + dims_ext];
    }

    return off;
}

inline dims_t get_reduced_shape(gsl::span<const size_t> in_shape,
                                gsl::span<const size_t> axis, bool keep_dims) {
    dims_t shape(in_shape.size() - (keep_dims ? 0 : axis.size()));
    auto count = 0;
    for (size_t i = 0; i < in_shape.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < axis.size(); j++) {
            if (i == axis[j]) {
                found = true;
                break;
            }
        }
        if (!found) {
            shape[count++] = in_shape[i];
        } else {
            if (keep_dims)
                shape[count++] = 1;
        }
    }
    return shape;
}
