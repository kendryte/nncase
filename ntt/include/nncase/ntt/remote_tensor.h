
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
#define PREFIX __device__
#else
#define PREFIX
#endif

PREFIX extern nncase::ntt::tensor<uintptr_t[2], nncase::ntt::distributed::topology_shape_t> global_local_data_ptr;

PREFIX extern nncase::ntt::tensor<uintptr_t, nncase::ntt::distributed::topology_shape_t> global_local_rdata_ptr;

namespace nncase::ntt::distributed {
template <class T, class Shape, topology Scope, class Strides>
class remote_tensor_view;

template <class T, IsFixedDims Shape, topology Scope, IsFixedDims Strides>
class remote_tensor_view<T, Shape, Scope, Strides>
    : public tensor_view<T, Shape, Strides> {
  public:
    using tensor_type = tensor_view<T, Shape, Strides>;
    using tensor_type::tensor_type;

    static auto get_start_offset(program_ids_t<Scope> local_program_ids,
                                 program_ids_t<Scope> remote_program_ids,
                                 const T *local_address) {
        auto start = global_local_data_ptr(local_program_ids)[0];
        auto end = global_local_data_ptr(local_program_ids)[1];
        auto remote_address = global_local_data_ptr(remote_program_ids)[0];
        if ((uintptr_t)local_address < start ||
            (uintptr_t)local_address >= end) {
            start = global_local_rdata_ptr(local_program_ids);
            remote_address = global_local_rdata_ptr(remote_program_ids);
        }

        return local_address - (T *)start + (T *)remote_address;
    }

    static auto create(program_ids_t<Scope> local_program_ids,
                       program_ids_t<Scope> remote_program_ids,
                       const T *local_address) noexcept {
        return remote_tensor_view<T, Shape, Scope, Strides>(
            std::span<T, tensor_type::shape()[0] * tensor_type::strides()[0]>(
                get_start_offset(local_program_ids, remote_program_ids,
                                 local_address),
                tensor_type::shape()[0] * tensor_type::strides()[0]));
    };

    tensor_type &tensor() noexcept { return static_cast<tensor_type &>(*this); }

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

    static auto create(program_ids_t<Scope> local_program_ids,
                       program_ids_t<Scope> remote_program_ids,
                       const T *local_address, Shape shape,
                       Strides strides) noexcept {
        return remote_tensor_view<T, Shape, Scope, Strides>(
            std::span<T, tensor_type::shape()[0] * tensor_type::strides()[0]>(
                get_start_offset(local_program_ids, remote_program_ids,
                                 local_address),
                tensor_type::shape()[0] * tensor_type::strides()[0]),
            shape, strides);
    };

    tensor_type &tensor() noexcept { return static_cast<tensor_type &>(*this); }

    const tensor_type &tensor() const noexcept {
        return static_cast<const tensor_type &>(*this);
    }
};
} // namespace nncase::ntt::distributed
