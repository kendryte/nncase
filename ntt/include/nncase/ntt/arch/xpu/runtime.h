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
#include "../../runtime.h"
#include "topology.h"
#include <cstdarg>

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace nncase::ntt::runtime {
struct xpu_block_entry_params_t {
    size_t tdim;
    size_t bdim;
    size_t ddim;
    size_t cdim;
    size_t bid;
    size_t did;
    size_t cid;
    size_t xpu_id_offset;
    std::byte *const *inouts;
    const std::byte *rdata;
};

struct xpu_thread_context_t {
    size_t tid;
    size_t bid;
    size_t did;
    size_t cid;

    static xpu_thread_context_t &current() noexcept;
};

extern size_t tdim;
extern size_t bdim;
extern size_t ddim;
extern size_t cdim;
} // namespace nncase::ntt::runtime
