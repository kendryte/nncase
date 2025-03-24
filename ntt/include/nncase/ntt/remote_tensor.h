
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
#if defined(NNCASE_XPU_MODULE)
#include "nncase/ntt/arch/xpu/topology.h"
#else
#include "nncase/ntt/arch/cpu/topology.h"
#endif

#include "tensor.h"

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
__device__
#endif
extern nncase::ntt::tensor<
    uint8_t *,
    nncase::ntt::ranked_shape<(
        size_t)nncase::ntt::distributed::topology::count__>,
    nncase::ntt::ranked_strides<(
        size_t)nncase::ntt::distributed::topology::count__>>
        global_local_data_ptr;

namespace nncase::ntt::distributed {
template <class T, class Shape, topology Scope, class Strides>
class remote_tensor_view;

template <class T, IsFixedDims Shape, topology Scope, IsFixedDims Strides>
class remote_tensor_view<T, Shape, Scope, Strides> : public tensor_view<T, Shape, Strides> {
  private:
    T *address_;

  public:
    using tensor_type = tensor_view<T, Shape, Strides>;
    using tensor_type::tensor_type;

    static auto create(program_ids_t<Scope> program_ids,
                                     size_t local_address) noexcept {
        return remote_tensor_view<T, Shape, Scope, Strides>(
            std::span<T, tensor_type::shape()[0] * tensor_type::strides()[0]>(local_address + (T *)global_local_data_ptr(program_ids),
                           tensor_type::shape()[0] * tensor_type::strides()[0]));
    };

    tensor_type &tensor() noexcept {
        return static_cast<tensor_type &>(*this);
    }

    const tensor_type &tensor() const noexcept {
        return static_cast<const tensor_type &>(*this);
    }
};

template <class T, class Shape, topology Scope, class Strides>
class remote_tensor_view : public tensor_view<T, Shape, Strides> {
  private:
    T *address_;

  public:
    using tensor_type = tensor_view<T, Shape, Strides>;
    using tensor_type::tensor_type;

    static auto create(program_ids_t<Scope> program_ids,
                                     size_t local_address, Shape shape, Strides strides) noexcept {
        return remote_tensor_view<T, Shape, Scope, Strides>(
            std::span<T, tensor_type::shape()[0] * tensor_type::strides()[0]>(local_address + (T *)global_local_data_ptr(program_ids),
                           tensor_type::shape()[0] * tensor_type::strides()[0]), shape, strides);
    };

    tensor_type &tensor() noexcept {
        return static_cast<tensor_type &>(*this);
    }

    const tensor_type &tensor() const noexcept {
        return static_cast<const tensor_type &>(*this);
    }
};
} // namespace nncase::ntt::distributed
