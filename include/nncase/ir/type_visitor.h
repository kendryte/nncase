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
#include "type.h"
#include <unordered_map>

namespace nncase::ir {

#define TYPE_FUNCTOR_DEFAULT                                                   \
    { return default_visit_type(t); }

#define TYPE_FUNCTOR_DISPATCH(type_t)                                          \
    if (auto n = t.as<type_t>())                                               \
    return visit_type(*n)

template <class R> class type_functor {
  public:
    virtual ~type_functor() = default;

    R operator()(const type &t) { return visit_type(t); }

    virtual R default_visit_type(const type &t) {
        throw std::runtime_error("Undispatched type: " +
                                 std::string(t->runtime_kind().name));
    }

    virtual R visit_type(const type &t) {
        assert(!t.empty());

        TYPE_FUNCTOR_DISPATCH(any_type);
        TYPE_FUNCTOR_DISPATCH(invalid_type);
        TYPE_FUNCTOR_DISPATCH(tensor_type);
        TYPE_FUNCTOR_DISPATCH(tuple_type);
        return default_visit_type(t);
    }

    virtual R visit_type(const any_type &t) TYPE_FUNCTOR_DEFAULT;
    virtual R visit_type(const invalid_type &t) TYPE_FUNCTOR_DEFAULT;
    virtual R visit_type(const tensor_type &t) TYPE_FUNCTOR_DEFAULT;
    virtual R visit_type(const tuple_type &t) TYPE_FUNCTOR_DEFAULT;
};

#undef TYPE_FUNCTOR_DEFAULT
#undef TYPE_FUNCTOR_DISPATCH
} // namespace nncase::ir
