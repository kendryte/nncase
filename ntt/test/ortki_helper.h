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
#include <cinttypes>
#include <ortki/c_api.h>
#include <ortki/operators.h>
#include <string>
#include <vector>

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

    // Helper to build an Ort tensor from any NTT tensor/vector instance.
    auto &src = tensor;
    auto ort_type = primitive_type2ort_type<T>();
    auto rank = src.shape().rank();
    std::vector<int64_t> shape_int64(rank);
    for (size_t i = 0; i < rank; i++)
        shape_int64[i] = src.shape()[i];

    auto ort_tensor =
        make_tensor_empty(ort_type, shape_int64.data(), rank);
    size_t bytes = 0;
    auto *dst = static_cast<T *>(tensor_buffer(ort_tensor, &bytes));
    [[maybe_unused]] const size_t total = src.shape().length();
    assert(bytes == total * sizeof(T));
    size_t linear_index = 0;
    ntt::apply(src.shape(), [&](auto index) {
        dst[linear_index++] = src(index);
    });
    return ort_tensor;
}

template <ntt::TensorOfVector TTensor>
ortki::OrtKITensor *ntt2ort(TTensor &tensor) {
    using vec_type = typename std::decay_t<TTensor>::element_type;
    auto RankDim = vec_type::rank();
    using vec_elem_type = ntt::element_or_scalar_t<vec_type>;
    auto ort_type = primitive_type2ort_type<vec_elem_type>();
    auto r1 = tensor.shape().rank();
    auto r2 = r1 + RankDim;
    std::vector<size_t> v(r2, 0);
    for (size_t i = 0; i < r1; i++)
        v[i] = tensor.shape()[i];
    for (size_t i = r1; i < r2; i++)
        v[i] = vec_type::shape()[i - r1];

    const int64_t *shape = reinterpret_cast<const int64_t *>(v.data());
    auto ort_tensor = make_tensor_empty(ort_type, shape, r2);
    size_t bytes = 0;
    auto *buffer_ptr = static_cast<vec_elem_type *>(tensor_buffer(ort_tensor, &bytes));
    [[maybe_unused]] const size_t total = tensor.shape().length() * vec_type::size();
    assert(bytes == total * sizeof(vec_elem_type));
    size_t linear_index = 0;
    ntt::apply(tensor.shape(), [&](auto tindex) {
        const auto &vec_src = tensor(tindex);
        ntt::apply(vec_src.shape(), [&](auto vindex) {
            buffer_ptr[linear_index++] = vec_src(vindex);
        });
    });

    return ort_tensor;
}

template <ntt::TensorOrVector TTensor>
void ort2ntt(ortki::OrtKITensor *ort_tensor, TTensor &ntt_tensor) {
    size_t size = 0;
    using element_type = ntt::element_or_scalar_t<TTensor>;
    auto ort_ptr = (const element_type *)tensor_buffer(ort_tensor, &size);
    assert(tensor_length(ort_tensor) == ntt_tensor.shape().length());
    ntt::apply(ntt_tensor.shape(), [&](auto tindex) {
        ntt_tensor(tindex) = *ort_ptr++;
    });
}

template <ntt::TensorOfVector TTensor>
void ort2ntt(ortki::OrtKITensor *ort_tensor, TTensor &ntt_tensor) {
    using vec_type = typename TTensor::element_type;
    assert(tensor_length(ort_tensor) ==
           ntt_tensor.shape().length() * vec_type::size());

    using vec_elem_type = typename vec_type::element_type;
    size_t size = 0;
    const vec_elem_type *ort_ptr = static_cast<const vec_elem_type *>(tensor_buffer(ort_tensor, &size));

    ntt::apply(ntt_tensor.shape(), [&](auto tindex) {
        auto &vec_dst = ntt_tensor(tindex);
        ntt::apply(ntt_tensor(tindex).shape(), [&](auto vindex) {
            vec_dst(vindex) = *ort_ptr++;
        });
    });
}
// template <ntt::TensorOfVector TTensor>
//     requires(TTensor::element_type::rank() == 1)
// void ort2ntt(ortki::OrtKITensor *ort_tensor, TTensor &ntt_tensor) {
//     size_t size = 0;
//     void *ort_ptr = tensor_buffer(ort_tensor, &size);
//     assert(tensor_length(ort_tensor) == ntt_tensor.size() *
//     TTensor::element_type::template lane<0>());
//     memcpy(ntt_tensor.elements().data(), ort_ptr, size);
// }

void print_ort_shape(ortki::OrtKITensor *ort_tensor) {
    auto rank = tensor_rank(ort_tensor);
    int64_t *shape = new int64_t[rank];
    tensor_shape(ort_tensor, shape);
    for (size_t i = 0; i < rank; ++i) {
        printf("%" PRIi64 " ", shape[i]);
    }
}

template<typename T>
constexpr size_t get_element_rank() {
    using element_type = typename std::decay_t<T>::element_type;
    if constexpr (ntt::Vector<element_type>) {
        return element_type::rank();
    } else {
        return 0;
    }
}

template<typename T>
void reshape_with_vector_alignment(ortki::OrtKITensor *&ort_tensor, const T &ntt_tensor, size_t higher_vector_rank) {
    assert(higher_vector_rank > 0);
    
    auto rank = ntt_tensor.shape().rank();
    std::vector<int64_t> new_shape_data;
    
    constexpr auto lower_vector_rank = get_element_rank<std::decay_t<T>>();
    
    new_shape_data.reserve(rank + higher_vector_rank);

    for (size_t i = 0; i < rank; ++i) { 
        new_shape_data.push_back(ntt_tensor.shape()[i]);
    }
    for (size_t i = 0; i < higher_vector_rank; ++i) {
        new_shape_data.push_back(1); 
    }
    if constexpr (lower_vector_rank > 0) {
        static_assert(lower_vector_rank == 1, "only support 1D vectors");
        using tensor_element_type = typename std::decay_t<T>::element_type;
        new_shape_data[rank+higher_vector_rank-1] = tensor_element_type::size();
    }

    int64_t reshape_shape[] = {static_cast<int64_t>(new_shape_data.size())};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(new_shape_data.data()),
                                   ort_type, reshape_shape, std::size(reshape_shape));
    ort_tensor = ortki_Reshape(ort_tensor, shape_tensor, 0);
}

template<typename T>
void reshape_for_outer_product(ortki::OrtKITensor *&ort_tensor, const T &ntt_tensor, bool is_lhs) {
    auto rank = ntt_tensor.shape().rank();
    std::vector<int64_t> new_shape_data;
    
    // Get vector length
    auto get_vlen = [&]() {
        if constexpr (get_element_rank<std::decay_t<T>>() > 0) {
            using tensor_element_type = typename std::decay_t<T>::element_type;
            return tensor_element_type::size();
        }
        return 1ul;
    };
    
    int64_t vlen = get_vlen();
    
    // Copy existing tensor shape
    for (size_t i = 0; i < rank; ++i) {
        new_shape_data.push_back(ntt_tensor.shape()[i]);
    }
    
    // Add outer product dimensions
    if (is_lhs) {
        // lhs: [..., lhs_vlen, 1]
        new_shape_data.push_back(vlen);
        new_shape_data.push_back(1);
    } else {
        // rhs: [..., 1, rhs_vlen]
        new_shape_data.push_back(1);
        new_shape_data.push_back(vlen);
    }
    
    int64_t reshape_shape[] = {static_cast<int64_t>(new_shape_data.size())};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(new_shape_data.data()),
                                   ort_type, reshape_shape, std::size(reshape_shape));
    ort_tensor = ortki_Reshape(ort_tensor, shape_tensor, 0);
}

//reshape means 
// 1. append dimension 1 at the last dimension which shoule be vector dimensions of ntt dimension
//    intput :lhs: (2 * 3 * 4) tensor of vector<2 * 4> rhs: (2 * 1 * 4) tensor of vector <4>
//    output :lhs  (2 * 3 * 4 * 2 * 4), rhs: (2 * 1 * 4 * "1" * 4)
// 2. for outer_product
//   input: lhs: 3 * 4 tensor of vector <8>  rhs: 3*4 tensor of vector <4>
//   output: lhs: 3 * 4 * 8 * 1, rhs: 3*4 * 1 * 4
//3. if need cast, cast the ort tensor into double
template <ntt::TensorOrVector TLhs, ntt::TensorOrVector TRhs>
auto convert_and_align_to_ort(TLhs &lhs, TRhs &rhs, bool need_cast = false,  bool for_outer_product = false) {
    auto ort_lhs = NttTest::ntt2ort(lhs);
    auto ort_rhs = NttTest::ntt2ort(rhs);

    constexpr size_t lhs_vector_rank = get_element_rank<TLhs>();
    constexpr size_t rhs_vector_rank = get_element_rank<TRhs>();
    
    if constexpr (lhs_vector_rank > rhs_vector_rank) {
        reshape_with_vector_alignment(ort_rhs, rhs, lhs_vector_rank);
    } else if constexpr (lhs_vector_rank < rhs_vector_rank) {
        reshape_with_vector_alignment(ort_lhs, lhs, rhs_vector_rank);
    }
    if (for_outer_product) {
        // For outer product, we need to reshape tensors for broadcasting
        // lhs should be reshaped to [..., lhs_vlen, 1]
        // rhs should be reshaped to [..., 1, rhs_vlen]
        // if element type is scalar, the *hs_vlen will be 1
        reshape_for_outer_product(ort_lhs, lhs, true);
        reshape_for_outer_product(ort_rhs, rhs, false);
    }
    

    if(need_cast){
        ort_lhs = ortki_Cast(ort_lhs,1,  ortki::DataType_DOUBLE);
        ort_rhs = ortki_Cast(ort_rhs,1,  ortki::DataType_DOUBLE);
    }

    return std::make_pair(ort_lhs, ort_rhs);
}

} // namespace NttTest
} // namespace nncase
