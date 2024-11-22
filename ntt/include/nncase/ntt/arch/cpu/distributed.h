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
#include "../../distributed.h"
#include "runtime.h"

namespace nncase::ntt::distributed {
template <> struct program_id_getter<topology::thread> {
    static size_t id() noexcept {
        return runtime::cpu_thread_context_t::current().tid;
    }
};

template <> struct program_id_getter<topology::block> {
    static size_t id() noexcept {
        return runtime::cpu_thread_context_t::current().bid;
    }
};

template <> struct program_id_getter<topology::chip> {
    static size_t id() noexcept {
        return runtime::cpu_thread_context_t::current().cid;
    }
};

inline size_t tid() noexcept { return program_id<topology::thread>(); }
inline size_t bid() noexcept { return program_id<topology::block>(); }
inline size_t cid() noexcept { return program_id<topology::chip>(); }

inline constexpr size_t tdim() noexcept {
    return program_dim(topology::thread);
}
inline constexpr size_t bdim() noexcept { return program_dim(topology::block); }
inline constexpr size_t cdim() noexcept { return program_dim(topology::chip); }
} // namespace nncase::ntt::distributed
