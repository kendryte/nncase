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
#include <array>
#include <cstdint>

template <class TScalar, size_t Lanes, class TVec>
TVec pack_elemt(const std::array<TScalar, Lanes> &arr);

#ifdef __ARM_NEON__
#include "arch/arm/pack_element.h"
#endif

#ifdef __AVX__
#include "arch/x86_64/pack_element.h"
#endif
