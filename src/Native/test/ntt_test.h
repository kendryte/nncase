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
#include <ortki/c_api.h>
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
#if defined __AVX2__
    __asm__ __volatile__("" : : : "memory");
    uint64_t r = __rdtsc();
    __asm__ __volatile__("" : : : "memory");
    return r;
#else
    return 0;
#endif
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
void init_tensor(ntt::tensor<ntt::vector<T, 8>, Shape, Stride> &tensor,
                 T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
    ntt::apply(tensor.shape(),
               [&](auto &index) { init_tensor(tensor(index), start, stop); });
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
ortki::OrtKITensor *ntt2ort(ntt::tensor<T, Shape, Stride> &tensor) {
    void *buffer = reinterpret_cast<void *>(tensor.elements().data());

    ortki::DataType ort_type = ortki::DataType_FLOAT;
    if (std::is_same_v<T, int8_t>)
        ort_type = ortki::DataType_INT8;
    else if (std::is_same_v<T, int16_t>)
        ort_type = ortki::DataType_INT16;
    else if (std::is_same_v<T, int32_t>)
        ort_type = ortki::DataType_INT32;
    else if (std::is_same_v<T, int64_t>)
        ort_type = ortki::DataType_INT64;
    else if (std::is_same_v<T, uint8_t>)
        ort_type = ortki::DataType_UINT8;
    else if (std::is_same_v<T, uint16_t>)
        ort_type = ortki::DataType_UINT16;
    else if (std::is_same_v<T, uint32_t>)
        ort_type = ortki::DataType_UINT32;
    else if (std::is_same_v<T, uint64_t>)
        ort_type = ortki::DataType_UINT64;
    else if (std::is_same_v<T, float>)
        ort_type = ortki::DataType_FLOAT;
    else if (std::is_same_v<T, double>)
        ort_type = ortki::DataType_DOUBLE;
    else {
        std::cerr << "unsupported data type" << std::endl;
        std::abort();
    }

    auto rank = tensor.shape().rank();
    std::vector<size_t> v(rank);
    for (size_t i = 0; i < rank; i++)
        v[i] = tensor.shape()[i];

    const int64_t *shape = reinterpret_cast<const int64_t *>(v.data());
    return make_tensor(buffer, ort_type, shape, rank);
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
void ort2ntt(ortki::OrtKITensor *ort_tensor,
             ntt::tensor<T, Shape, Stride> &ntt_tensor) {
    size_t size = 0;
    void *ort_ptr = tensor_buffer(ort_tensor, &size);
    assert(tensor_length(ort_tensor) == ntt_tensor.shape().length());
    memcpy((void *)ntt_tensor.elements().data(), ort_ptr, size);
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
bool compare_tensor(ntt::tensor<T, Shape, Stride> &lhs,
                    ntt::tensor<T, Shape, Stride> &rhs,
                    double threshold = 0.999f) {
    if (lhs.shape().rank() != rhs.shape().rank()) {
        return false;
    }

    for (size_t i = 0; i < lhs.shape().rank(); i++)
        if (lhs.shape()[i] != rhs.shape()[i])
            return false;

    std::vector<double> v1;
    std::vector<double> v2;
    v1.reserve(lhs.shape().length());
    v2.reserve(rhs.shape().length());

    bool pass = true;
    nncase::ntt::apply(lhs.shape(), [&](auto index) {
        auto lvalue = lhs(index);
        auto rvalue = rhs(index);
        v1.push_back(static_cast<double>(lvalue));
        v2.push_back(static_cast<double>(rvalue));

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
} // namespace NttTest
} // namespace nncase