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
#include "nncase/ntt/tensor_traits.h"
#include <assert.h>
#include <ortki/c_api.h>
#include <string>
#include <ortki/operators.h>
#include <cinttypes>

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
    else if (std::is_same_v<T, half>)
        ort_type = ortki::DataType_FLOAT16;
    else if (std::is_same_v<T, double>)
        ort_type = ortki::DataType_DOUBLE;
    else if (std::is_same_v<T, bool>)
        ort_type = ortki::DataType_BOOL;
    else if (std::is_same_v<T, bfloat16>)
        ort_type = ortki::DataType_BFLOAT16;
    else {
        std::cerr << __FUNCTION__ << ": unsupported data type" << std::endl;
        std::abort();
    }

    return ort_type;
}

template <ntt::TensorOrVector TTensor>
ortki::OrtKITensor *ntt2ort(TTensor &tensor) {
    using T = typename std::decay_t<TTensor>::element_type;
    void *buffer;
    if constexpr (ntt::Vector<TTensor>) {
        buffer = &tensor.buffer();
    } else {
        buffer = tensor.elements().data();
    }
    auto ort_type = primitive_type2ort_type<T>();
    auto rank = tensor.shape().rank();
    std::vector<size_t> v(rank);
    for (size_t i = 0; i < rank; i++)
        v[i] = tensor.shape()[i];

    const int64_t *shape = reinterpret_cast<const int64_t *>(v.data());
    return make_tensor(buffer, ort_type, shape, rank);
}

template <ntt::TensorOfVector TTensor>
ortki::OrtKITensor *ntt2ort(TTensor &tensor) {
    using T = typename std::decay_t<TTensor>::element_type;
    auto N = T::size();
    auto RankDim = T::rank();
    using ElemType = ntt::element_or_scalar_t<T>;
    void *buffer = reinterpret_cast<void *>(tensor.elements().data());
    auto ort_type = primitive_type2ort_type<ElemType>();
    auto r1 = tensor.shape().rank();
    auto r2 = r1 + RankDim;
    std::vector<size_t> v(r2, N);
    for (size_t i = 0; i < r1; i++)
        v[i] = tensor.shape()[i];

    const int64_t *shape = reinterpret_cast<const int64_t *>(v.data());
    return make_tensor(buffer, ort_type, shape, r2);
}

template <ntt::TensorOrVector TTensor>
void ort2ntt(ortki::OrtKITensor *ort_tensor, TTensor &ntt_tensor) {
    size_t size = 0;
    void *ort_ptr = tensor_buffer(ort_tensor, &size);
    using element_type = ntt::element_or_scalar_t<TTensor>;
    if constexpr (ntt::Vector<element_type>) {
        assert(tensor_length(ort_tensor) ==
               ntt_tensor.shape().length() * element_type::size());
    } else {
        assert(tensor_length(ort_tensor) == ntt_tensor.shape().length());
    }
    if constexpr (ntt::Vector<TTensor>) {
        memcpy(&ntt_tensor.buffer(), ort_ptr, size);
    } else {
        memcpy(ntt_tensor.elements().data(), ort_ptr, size);
    }
}

template <ntt::TensorOfVector TTensor>
    requires(TTensor::element_type::rank() == 1)
void ort2ntt(ortki::OrtKITensor *ort_tensor, TTensor &ntt_tensor) {
    size_t size = 0;
    void *ort_ptr = tensor_buffer(ort_tensor, &size);
    assert(tensor_length(ort_tensor) == ntt_tensor.size() * TTensor::element_type::template lane<0>());
    memcpy(ntt_tensor.elements().data(), ort_ptr, size);
}

void print_ort_shape(ortki::OrtKITensor *ort_tensor) {
    auto rank = tensor_rank(ort_tensor);
    int64_t *shape = new int64_t[rank];
    tensor_shape(ort_tensor, shape);
    for(size_t i=0; i < rank; ++i)
    {
        printf(PRIi64, shape[i]);
    }
}

template <ntt::TensorOrVector TLhs, ntt::TensorOrVector TRhs>
auto convert_and_align_to_ort(TLhs& lhs, TRhs& rhs) {
    auto ort_lhs = NttTest::ntt2ort(lhs);
    auto ort_rhs = NttTest::ntt2ort(rhs);

    constexpr bool lhs_is_vec = ntt::Vector<typename TLhs::element_type>;
    constexpr bool rhs_is_vec = ntt::Vector<typename TRhs::element_type>;
    //TODO: deal with the case that 2D vector and 1D vector
    auto reshape_op = [](auto &orttensor_to_append,
                         const auto &ntttensor_to_append) {
        auto rank = ntttensor_to_append.shape().rank();
        std::vector<int64_t> new_shape_data;
        new_shape_data.reserve(rank + 1);
        for (size_t i = 0; i < rank; ++i) {
            new_shape_data.push_back(ntttensor_to_append.shape()[i]);
        }
        new_shape_data.push_back(1);
        int64_t reshape_shape[] = {
            static_cast<int64_t>(new_shape_data.size())};
        auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
        auto shape_tensor =
            make_tensor(reinterpret_cast<void *>(new_shape_data.data()),
                        ort_type, reshape_shape, std::size(reshape_shape));
        orttensor_to_append =
            ortki_Reshape(orttensor_to_append, shape_tensor, 0);
    };

    if constexpr (lhs_is_vec && !rhs_is_vec) {
        reshape_op(ort_rhs, rhs);
    } else if constexpr (!lhs_is_vec && rhs_is_vec) {
        reshape_op(ort_lhs, lhs);
    }
    return std::make_pair(ort_lhs, ort_rhs);
}
} // namespace NttTest
} // namespace nncase