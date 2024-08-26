/* Copyright 2019-2024 Canaan Inc.
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
#include "nncase/ntt/apply.h"
#include "nncase/ntt/ntt.h"
#include "nncase/ntt/shape.h"
#include <assert.h>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

#ifdef __AVX2__
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#define ULP_SIZE 250000
#ifndef ULP_SIZE
#define ULP_SIZE 10000
#endif

#ifndef CPU_FREQUENCY_MHZ
#define CPU_FREQUENCY_MHZ 1600
#endif

#ifndef CLOCK_SOURCE_FREQUENCY_MHZ
#define CLOCK_SOURCE_FREQUENCY_MHZ 27
#endif

namespace nncase {
namespace NttTest {

__inline__ uint64_t get_cpu_cycle(void) {
    uint64_t cycles = 0;
#if defined __x86_64
    __asm__ __volatile__("" : : : "memory");
    cycles = __rdtsc();
    __asm__ __volatile__("" : : : "memory");
#elif defined __riscv
    uint64_t time = 0;
    asm volatile("rdtime %0" : "=r"(time));
    cycles = time * CPU_FREQUENCY_MHZ / CLOCK_SOURCE_FREQUENCY_MHZ;
#endif
    return cycles;
}

template <typename T, class TTensor>
void init_tensor(TTensor &tensor, T start = static_cast<T>(0),
                 T stop = static_cast<T>(1)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    if (std::is_same_v<T, int8_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<int8_t>(dis(gen));
        });
    } else if (std::is_same_v<T, int16_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<int16_t>(dis(gen));
        });
    } else if (std::is_same_v<T, int32_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<int32_t>(dis(gen));
            // std::cout << "index(";
            // for (size_t i = 0; i < index.rank(); i++)
            //     std::cout << index[i] << " ";
            // std::cout << ") = " << tensor(index) << std::endl;
        });
    } else if (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<int64_t>(dis(gen));
        });
    } else if (std::is_same_v<T, uint8_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<uint8_t>(dis(gen));
        });
    } else if (std::is_same_v<T, uint16_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<uint16_t>(dis(gen));
        });
    } else if (std::is_same_v<T, uint32_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<uint32_t>(dis(gen));
        });
    } else if (std::is_same_v<T, uint64_t>) {
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<uint64_t>(dis(gen));
        });
    } else if (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<float>(dis(gen));
            // std::cout << "index(";
            // for (size_t i = 0; i < index.rank(); i++)
            //     std::cout << index[i] << " ";
            // std::cout << ") = " << tensor(index) << std::endl;
        });
    } else if (std::is_same_v<T, double>) {
        std::uniform_real_distribution<double> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<double>(dis(gen));
        });
    } else if (std::is_same_v<T, bool>) {
        std::uniform_real_distribution<double> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<double>(dis(gen)) >= 0.5;
        });
    } else {
        std::cerr << "unsupported data type" << std::endl;
        std::abort();
    }
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>, size_t N>
void init_tensor(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}

template <ntt::IsTensor TTensor>
bool compare_tensor(TTensor &lhs, TTensor &rhs, double threshold = 0.999f) {
    using T = typename std::decay_t<TTensor>::element_type;
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);
        v1.push_back(lvalue);
        v2.push_back(rvalue);
        if (lvalue != rvalue) {
            // std::cout << "index = (";
            // for (size_t i = 0; i < index.rank(); i++)
            //     std::cout << index[i] << " ";
            // std::cout << "): lhs = " << lvalue << ", rhs = " << rvalue
            //           << std::endl;
            pass = false;
        }
    });

    if (!pass) {
        double dotProduct =
            std::inner_product(v1.begin(), v1.end(), v2.begin(), (double)0.0);
        double norm1 = std::sqrt(
            std::inner_product(v1.begin(), v1.end(), v1.begin(), (double)0.0));
        double norm2 = std::sqrt(
            std::inner_product(v2.begin(), v2.end(), v2.begin(), (double)0.0));
        double cosine_similarity = dotProduct / (norm1 * norm2);
        pass = cosine_similarity > threshold;
        if (!pass)
            std::cerr << "cosine_similarity = " << cosine_similarity
                      << std::endl;
    }
    return pass;
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>, size_t N>
bool compare_tensor(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &lhs,
                    ntt::tensor<ntt::vector<T, N>, Shape, Stride> &rhs,
                    double threshold = 0.999f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);

        nncase::ntt::apply(lvalue.shape(), [&](auto idx) {
            v1.push_back(lvalue(idx));
            v2.push_back(rvalue(idx));
            if (lvalue(idx) != rvalue(idx)) {
                // std::cout << "index = (";
                // for (size_t i = 0; i < index.rank(); i++)
                //     std::cout << index[i] << " ";
                // std::cout << "): lhs = " << lvalue(idx)
                //           << ", rhs = " << rvalue(idx) << std::endl;
                pass = false;
            }
        });
    });

    if (!pass) {
        double dotProduct =
            std::inner_product(v1.begin(), v1.end(), v2.begin(), (double)0.0);
        double norm1 = std::sqrt(
            std::inner_product(v1.begin(), v1.end(), v1.begin(), (double)0.0));
        double norm2 = std::sqrt(
            std::inner_product(v2.begin(), v2.end(), v2.begin(), (double)0.0));
        double cosine_similarity = dotProduct / (norm1 * norm2);
        pass = cosine_similarity > threshold;
        if (!pass)
            std::cerr << "cosine_similarity = " << cosine_similarity
                      << std::endl;
    }
    return pass;
}

template <typename T> T ulp(T x) {
    x = std::fabs(x);
    if (std::isfinite(x)) {
        T lower = std::nextafter(x, static_cast<T>(-1.0));
        return x - lower;
    }
    return x;
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
bool compare_ulp(ntt::tensor<T, Shape, Stride> &lhs,
                 ntt::tensor<T, Shape, Stride> &rhs, double threshold = 0.5f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    double max_ulp_error = 0.f;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);
        auto ulp_error = std::abs(lvalue - rvalue) / ulp(rvalue);
        max_ulp_error = ulp_error > max_ulp_error ? ulp_error : max_ulp_error;
    });

    if (max_ulp_error > threshold) {
        std::cout << "ulp threshold = " << threshold
                  << ", max_ulp_error = " << max_ulp_error << std::endl;

        pass = false;
    }
    // std::cout << "ulp threshold = " << threshold
    //           << ", max_ulp_error = " << max_ulp_error << std::endl;
    return pass;
}

template <typename T, size_t N, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
bool compare_ulp(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &lhs,
                 ntt::tensor<ntt::vector<T, N>, Shape, Stride> &rhs,
                 double threshold = 0.5f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<T> v1;
    std::vector<T> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    double max_ulp_error = 0.f;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);

        nncase::ntt::apply(lvalue.shape(), [&](auto idx) {
            auto ulp_error =
                std::abs(lvalue(idx) - rvalue(idx)) / ulp((T)rvalue(idx));
            if (ulp_error > max_ulp_error)
                std::cout << "lvalue(idx) = " << lvalue(idx)
                          << ", rvalue(idx) = " << rvalue(idx)
                          << ", ulp = " << ulp((T)rvalue(idx))
                          << ", ulp_error = " << ulp_error
                          << ", max_ulp_error = " << max_ulp_error << std::endl;
            max_ulp_error =
                ulp_error > max_ulp_error ? ulp_error : max_ulp_error;
        });
    });

    if (max_ulp_error > threshold) {
        std::cout << "ulp threshold = " << threshold
                  << ", max_ulp_error = " << max_ulp_error << std::endl;

        pass = false;
    }
    std::cout << "ulp threshold = " << threshold
              << ", max_ulp_error = " << max_ulp_error << std::endl;
    return pass;
}

} // namespace NttTest
} // namespace nncase