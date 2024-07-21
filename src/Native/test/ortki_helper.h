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
#include <ortki/c_api.h>
#include <string>

namespace nncase {
namespace NttTest {

template <typename T> ortki::DataType primitive_type2ort_type() {
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

    return ort_type;
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>>
ortki::OrtKITensor *ntt2ort(ntt::tensor<T, Shape, Stride> &tensor) {
    void *buffer = reinterpret_cast<void *>(tensor.elements().data());
    auto ort_type = primitive_type2ort_type<T>();
    auto rank = tensor.shape().rank();
    std::vector<size_t> v(rank);
    for (size_t i = 0; i < rank; i++)
        v[i] = tensor.shape()[i];

    const int64_t *shape = reinterpret_cast<const int64_t *>(v.data());
    return make_tensor(buffer, ort_type, shape, rank);
}

template <typename T, typename Shape,
          typename Stride = ntt::default_strides_t<Shape>, size_t N>
ortki::OrtKITensor *
ntt2ort(ntt::tensor<ntt::vector<T, N>, Shape, Stride> &tensor) {
    void *buffer = reinterpret_cast<void *>(tensor.elements().data());
    auto ort_type = primitive_type2ort_type<T>();
    auto rank1 = tensor.shape().rank();
    auto rank = rank1 + tensor(0).shape().rank();
    std::vector<size_t> v(rank);
    for (size_t i = 0; i < rank1; i++)
        v[i] = tensor.shape()[i];

    for (size_t i = rank1; i < rank; i++)
        v[i] = tensor(0).shape()[i];

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
          typename Stride = ntt::default_strides_t<Shape>, size_t N>
void ort2ntt(ortki::OrtKITensor *ort_tensor,
             ntt::tensor<ntt::vector<T, N>, Shape, Stride> &ntt_tensor) {
    size_t size = 0;
    void *ort_ptr = tensor_buffer(ort_tensor, &size);
    assert(tensor_length(ort_tensor) == ntt_tensor.shape().length());
    memcpy((void *)ntt_tensor.elements().data(), ort_ptr, size);
}
} // namespace NttTest
} // namespace nncase