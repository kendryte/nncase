/* Copyright 2019-2021 Canaan Inc.
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
#include "nncase/ntt/shape.h"
#include "tensor.h"
#include "tensor_traits.h"

namespace nncase::ntt::tensor_ops {
template <Tensor TTensor> struct tload {
    using T = typename TTensor::element_type;

    constexpr TTensor operator()(const T *src) const noexcept {
        TTensor vec;
        std::copy(src, src + vec.size(), vec.buffer().data());
        return vec;
    }
};

template <Tensor TTensor> struct tload_scalar {
    using T = typename TTensor::element_type;

    constexpr TTensor operator()(const T &value) const noexcept {
        TTensor vec;
        std::fill_n(vec.buffer().data(), vec.size(), value);
        return vec;
    }
};
} // namespace nncase::ntt::tensor_ops

namespace nncase::ntt {
template <ScalarOrVector T, Shape TShape, Strides TStrides, bool IsView>
basic_tensor<T, TShape, TStrides, IsView>
basic_tensor<T, TShape, TStrides, IsView>::from_scalar(T value) noexcept {
    return tensor_ops::tload_scalar<basic_tensor<T, TShape, TStrides, false>>()(
        value);
}

template <Tensor TTensor, Scalar T>
constexpr TTensor tload(const T *src) noexcept {
    return tensor_ops::tload<TTensor>()(src);
}
} // namespace nncase::ntt
