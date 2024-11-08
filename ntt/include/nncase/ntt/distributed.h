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
#include "tensor.h"
#include <cstddef>

namespace nncase::ntt {
template <size_t Axis> struct program_id_getter {
    static size_t id() noexcept;
    static size_t dim() noexcept;
};

template <size_t Axis> size_t program_id() noexcept {
    return program_id_getter<Axis>::id();
}

template <size_t Axis> size_t program_dim() noexcept {
    return program_id_getter<Axis>::dim();
}

inline size_t tid() noexcept { return program_id<0>(); }
inline size_t tdim() noexcept { return program_dim<0>(); }
inline size_t bid() noexcept { return program_id<1>(); }
inline size_t bdim() noexcept { return program_dim<1>(); }
} // namespace nncase::ntt
