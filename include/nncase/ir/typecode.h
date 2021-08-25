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
#include <nncase/runtime/datatypes.h>

namespace nncase::ir
{
template <class T>
class expr_t;

class function_node;
using function = expr_t<function_node>;

enum class typecode_t : uint8_t
{
#define DEFINE_TYPECODE(id, t, name, value) id = value,
#include "typecode.def"
#undef DEFINE_TYPECODE
};

namespace detail
{
    template <typecode_t Type>
    struct typecode_to_cpp_type
    {
    };

    template <class T>
    struct cpp_type_to_typecode
    {
    };

    template <>
    struct cpp_type_to_typecode<std::byte>
    {
        static constexpr typecode_t type = typecode_t::uint8;
    };

#define DEFINE_TYPECODE(id, t, name, value)                \
    template <>                                            \
    struct typecode_to_cpp_type<typecode_t::id>            \
    {                                                      \
        using type = t;                                    \
    };                                                     \
    template <>                                            \
    struct cpp_type_to_typecode<t>                         \
    {                                                      \
        static constexpr typecode_t type = typecode_t::id; \
    };
#include "typecode.def"
#undef DEFINE_TYPECODE
}

template <class T>
constexpr typecode_t to_typecode() noexcept
{
    return detail::cpp_type_to_typecode<T>::type;
}

template <typecode_t Type>
using to_cpp_type_t = typename detail::typecode_to_cpp_type<Type>::type;

}
