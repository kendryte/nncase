#pragma once
#include <cmath>
#include <iostream>
#include <nncase/runtime/simple_types.h>
#include <string>

namespace nncase::runtime {
inline float dot(const float *v1, const float *v2, size_t size) {
    float ret = 0.f;
    for (size_t i = 0; i < size; i++) {
        ret += v1[i] * v2[i];
    }

    return ret;
}

inline float cosine(const float *v1, const float *v2, size_t size) {
    return dot(v1, v2, size) /
           ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
}

inline void dump_shape(gsl::span<const size_t> shape) {
    std::cout << "shape:";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << " ";
    }
    std::cout << "\n";
}
} // namespace nncase::runtime