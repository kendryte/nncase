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
#include "../../distributed/topology.h"
#include "runtime.h"
#include <barrier>
#include <cooperative_groups.h>

namespace nncase::ntt::distributed {
template <> struct program_id_getter<topology::thread> {
    __device__ static size_t id() noexcept { return threadIdx.x / warpSize; }
};

template <> struct program_id_getter<topology::block> {
    __device__ static size_t id() noexcept { return blockIdx.x; }
};

template <> struct program_id_getter<topology::chip> {
    __device__ static size_t id() noexcept {
        return runtime::cuda_thread_context_t::current().cid;
    }
};

inline __device__ size_t lane_id() noexcept {
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

inline __device__ size_t tid() noexcept {
    return program_id<topology::thread>();
}
inline __device__ size_t bid() noexcept {
    return program_id<topology::block>();
}
inline __device__ size_t cid() noexcept { return program_id<topology::chip>(); }

inline constexpr auto tdim() noexcept {
    return program_dim<topology::thread>();
}
inline constexpr auto bdim() noexcept { return program_dim<topology::block>(); }
inline constexpr auto cdim() noexcept { return program_dim<topology::chip>(); }

template <> class topology_synchronizer<topology::thread> {
  public:
    __device__ static void synchronize() noexcept { __syncthreads(); }
};

template <> class topology_synchronizer<topology::block> {
  public:
    __device__ static void synchronize() noexcept {
        cooperative_groups::grid_group g = cooperative_groups::this_grid();
        g.sync();
    }
};

template <> class topology_synchronizer<topology::chip> {
  public:
    __device__ static void synchronize() noexcept {}
};
} // namespace nncase::ntt::distributed
