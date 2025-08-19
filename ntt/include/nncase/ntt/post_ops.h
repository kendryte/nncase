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
#include "shape.h"
#include "tensor_traits.h"
#include <cstring>
#include <type_traits>

namespace nncase::ntt {
namespace post_ops_detail {
template <typename T> struct extract_op_type { using type = void; };

template <template <typename> class Template, typename Param>
struct extract_op_type<Template<Param>> {
    using type = Param;
};
} // namespace post_ops_detail

template <typename T> struct DefaultPostOp {
    constexpr T operator()(const T &a) const noexcept { return a; }
};
} // namespace nncase::ntt