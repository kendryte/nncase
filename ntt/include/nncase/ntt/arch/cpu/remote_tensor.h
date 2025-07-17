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
#include "../../distributed/remote_tensor.h"
#include "../../tensor.h"
#include "../../vector.h"

namespace nncase::ntt::distributed {
namespace detail {
extern decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape)) global_local_data_ptr;

extern decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape)) global_thread_local_rdata_ptr;

extern decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape)) global_block_local_rdata_ptr;

template <class T, topology RemoteScope, topology TensorScope,
          ScopedProgramIds<TensorScope> TLocalProgramIds,
          ScopedProgramIds<TensorScope> TRemoteProgramIds>
static auto get_remote_address(const TLocalProgramIds &local_program_ids,
                               const TRemoteProgramIds &remote_program_ids,
                               T *local_address) {
    auto start = global_local_data_ptr(local_program_ids)(0_dim);
    auto end = global_local_data_ptr(local_program_ids)(1_dim);
    auto remote_address = global_local_data_ptr(remote_program_ids)(0_dim);
    if ((uintptr_t)local_address < start || (uintptr_t)local_address >= end) {
        start = global_thread_local_rdata_ptr(local_program_ids)(0_dim);
        end = global_thread_local_rdata_ptr(local_program_ids)(1_dim);
        remote_address =
            global_thread_local_rdata_ptr(remote_program_ids)(0_dim);
        if ((uintptr_t)local_address < start ||
            (uintptr_t)local_address >= end) {
            start = global_block_local_rdata_ptr(local_program_ids)(0_dim);
            remote_address =
                global_block_local_rdata_ptr(remote_program_ids)(0_dim);
        }
    }

    return local_address - (T *)start + (T *)remote_address;
}
} // namespace detail

template <topology RemoteScope, topology TensorScope>
struct remote_tensor_constructor {
    template <class T, Shape TShape, Strides TStrides,
              ScopedProgramIds<TensorScope> TLocalProgramIds,
              ScopedProgramIds<TensorScope> TRemoteProgramIds>
    constexpr auto operator()(T *data, const TShape &shape,
                              const TStrides &strides,
                              const TLocalProgramIds &local_program_ids,
                              const TRemoteProgramIds &remote_program_ids) {
        auto remote_address =
            detail::get_remote_address<T, RemoteScope, TensorScope>(
                local_program_ids, remote_program_ids, data);
        return make_tensor_view_from_address(remote_address, shape, strides);
    }
};
} // namespace nncase::ntt::distributed
