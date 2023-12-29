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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <span>
#include <utility>

namespace nncase::ntt {
template <class U, class T, size_t Extent>
auto span_cast(std::span<T, Extent> span) {
    using return_type =
        std::conditional_t<Extent == std::dynamic_extent, std::span<U>,
                           std::span<U, Extent * sizeof(T) / sizeof(U)>>;
    return return_type(reinterpret_cast<U *>(span.data()),
                       span.size_bytes() / sizeof(U));
}
} // namespace nncase::ntt
