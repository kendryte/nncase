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
#include "../../distributed/topology.h"
#include "../../tensor.h"
#include "../../vector.h"
#include <cstdint>

#if defined(NNCASE_XPU_MODULE) && defined(SYS_MODE)
#define PREFIX __device__
#else
#define PREFIX
#endif

PREFIX extern decltype(nncase::ntt::make_tensor<
                       nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape)) global_local_data_ptr;

PREFIX extern decltype(nncase::ntt::make_tensor<uintptr_t>(
    nncase::ntt::distributed::topology_shape)) global_local_rdata_ptr;

namespace nncase::ntt::distributed {} // namespace nncase::ntt::distributed
