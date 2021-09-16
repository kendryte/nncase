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
#include "expr.h"
#include "function.h"
#include "ir_types.h"
#include "type.h"

namespace nncase::ir {
class NNCASE_API type_infer_context {
  public:
    virtual ~type_infer_context() = default;

    virtual type argument_type(const connector_info &parameter) = 0;
    virtual expr argument(const connector_info &parameter) = 0;

    template <class... TArgs>
    auto argument_types(const TArgs &...parameters)
        -> std::array<type, sizeof...(parameters)> {
        return {argument_type(parameters)...};
    }
};

/** @brief Run type inference on the function */
NNCASE_API bool infer_type(const function &func);

#define CHECK_TYPE(cond, reason)                                               \
    if (!(cond))                                                               \
        return invalid_type(reason);

#define CHECK_ARGUMENT_TYPE(name, t, reason)                                   \
    auto name##_t_raw = context.argument_type(name());                         \
    if (name##_t_raw.is_a<any_type>())                                         \
        return any_type();                                                     \
    auto name##_t_opt = name##_t_raw.as<t>();                                  \
    if (!name##_t_opt)                                                         \
        return invalid_type(reason);                                           \
    auto &name##_t = *name##_t_opt

#define CHECK_ARGUMENT_AS_TENSOR(name)                                         \
    CHECK_ARGUMENT_TYPE(name, tensor_type, #name "must be a tensor")

NNCASE_API type broadcast_type(std::initializer_list<tensor_type> inputs);

inline type broadcast_type(const tensor_type &lhs, const tensor_type &rhs) {
    return broadcast_type({lhs, rhs});
}
} // namespace nncase::ir
