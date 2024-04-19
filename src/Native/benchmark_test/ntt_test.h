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
#include "nncase/ntt/shape.h"
#include <assert.h>
#include <iostream>
#include <random>
#include <string>

#ifdef __AVX2__
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
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
    asm volatile("rdcycle %0" : "=r"(cycles));
#endif
    return cycles;
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
void init_tensor(ntt::tensor<T, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
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
          typename Stride = ntt::default_strides_t<Shape>>
void init_tensor(ntt::tensor<ntt::vector<T, 4>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
void init_tensor(ntt::tensor<ntt::vector<T, 8>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}
template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
void init_tensor(ntt::tensor<ntt::vector<T, 16>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
void init_tensor(ntt::tensor<ntt::vector<T, 32>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}

} // namespace NttTest
} // namespace nncase