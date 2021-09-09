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
#include "../object.h"
#include "call.h"
#include "constant.h"

namespace nncase::ir::F {
namespace detail {
template <class T> concept Scalar = requires {
    nncase::detail::cpp_type_to_datatype<T>::type;
};
} // namespace detail

class fexpr : public expr {
  public:
    template <Expr T> fexpr(T &&other) noexcept : expr(std::move(other)) {}
    template <Expr T> fexpr(const T &other) noexcept : expr(other) {}

    template <detail::Scalar T>
    fexpr(T scalar)
        : expr(constant(prim_type(to_datatype<T>()),
                        std::span<const T>(&scalar, 1))) {}
};
} // namespace nncase::ir::F
