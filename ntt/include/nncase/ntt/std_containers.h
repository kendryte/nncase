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
#include "compiler_defs.h"
#include <array>
#include <span>

#ifdef __CUDA_ARCH__
#include <cuda/std/array>
#include <cuda/std/span>
#endif

namespace nncase::ntt {
#ifdef __CUDA_ARCH__
using cuda::std::array;
using cuda::std::span;
#else
using std::array;
using std::span;
#endif
} // namespace nncase::ntt
