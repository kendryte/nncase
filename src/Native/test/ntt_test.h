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
#include <iostream>
#include <random>
#include <string>
#include <assert.h>
#include <ortki/c_api.h>
#include "nncase/ntt/apply.h"
#include "nncase/ntt/shape.h"

namespace nncase {
namespace NttTest {
    template <typename T, typename Shape, typename Stride = ntt::default_strides_t<Shape>>
    void init_tensor(ntt::tensor<T, Shape, Stride> &tensor, T start = static_cast<T>(0), T stop = static_cast<T>(1)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(start, stop);
        ntt::apply(tensor.shape(), [&](auto &index) {
            tensor(index) = static_cast<T>(dis(gen));
            // std::cout << "index(";
            // for (size_t i = 0; i < index.rank(); i++)
            //     std::cout << index[i] << " ";
            // std::cout << ") = " << tensor(index) << std::endl;
        });
    }

    template <typename T, typename Shape, typename Stride = ntt::default_strides_t<Shape>>
    ortki::OrtKITensor *
    ntt2ort(ntt::tensor<T, Shape, Stride> &tensor) {
        void *buffer = reinterpret_cast<void *>(tensor.buffer().data());

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

    template <typename T, typename Shape, typename Stride = ntt::default_strides_t<Shape>>
    void ort2ntt(ortki::OrtKITensor *ort_tensor,
                               ntt::tensor<T, Shape, Stride> &ntt_tensor) {
        size_t size = 0;
        void *ort_ptr = tensor_buffer(ort_tensor, &size);
        assert(tensor_length(ort_tensor) == ntt_tensor.shape().length());
        memcpy((void *)ntt_tensor.buffer().data(), ort_ptr, size);
    }

    template <typename T, typename Shape, typename Stride = ntt::default_strides_t<Shape>>
    bool compare_tensor(ntt::tensor<T, Shape, Stride> &lhs,
                        ntt::tensor<T, Shape, Stride> &rhs,
                        double threshold = 0.999f)
    {
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
                std::cout << "index = (";
                for(size_t i = 0; i < index.rank(); i++)
                    std::cout << index[i] << " ";
                std::cout << "): lhs = " << lhs(index) << ", rhs = " << rhs(index) << std::endl;
                pass = false;
            }
        });

        if (!pass)
        {
            double dotProduct = std::inner_product(v1.begin(), v1.end(), v2.begin(), (double)0.0);
            double norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), (double)0.0));
            double norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), (double)0.0));
            double cosine_similarity = dotProduct / (norm1 * norm2);
            pass = cosine_similarity > threshold;
            if (!pass)
                std::cerr << "cosine_similarity = " << cosine_similarity << std::endl;
        }
        return pass;
    }
}// namespace NttTest
} // namespace nncase