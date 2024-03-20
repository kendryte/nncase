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
#include <arm_neon.h>
#include <array>

inline float32x4_t pack_elemt(const std::array<float, 4> &vec) {
    return vld1q_f32(&vec[0]);
}

inline float32x2_t pack_elemt(const std::array<float, 2> &vec) {
    return vld1_f32(&vec[0]);
}
