
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
#include "sharding.h"
#include "tensor.h"

namespace nncase::ntt::distributed {
template <class T, class Shape, class Strides> class remote_tensor_view {
  public:
    static remote_tensor_view create(ranked_shape<topology_levels> program_ids,
                                     T *local_address) noexcept;
};
} // namespace nncase::ntt::distributed
