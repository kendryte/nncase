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
#include "primitive_ops.bfloat16.h"
#include "primitive_ops.float16.h"
#include "primitive_ops.float32.h"

namespace nncase::ntt::ops {
// load/store operation functors
template <> struct prefetch<prefetch_hint::l0, true> {
    void operator()(const void *ptr) const noexcept {
        _mm_prefetch(ptr, _MM_HINT_T0);
    }
};

template <> struct prefetch<prefetch_hint::l1, true> {
    void operator()(const void *ptr) const noexcept {
        _mm_prefetch(ptr, _MM_HINT_T1);
    }
};

template <> struct prefetch<prefetch_hint::l2, true> {
    void operator()(const void *ptr) const noexcept {
        _mm_prefetch(ptr, _MM_HINT_T2);
    }
};
} // namespace nncase::ntt::ops
