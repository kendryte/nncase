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
#include <map>
#include <string>

namespace nncase
{
constexpr std::string_view datatype_names(datatype_t dt)
{
    switch (dt)
    {
#define DEFINE_DATATYPE(id, t, name, value) \
    case dt_##id:                           \
        return #name;
#include <nncase/runtime/datatypes.def>
#undef DEFINE_DATATYPE
    default:
        throw std::invalid_argument("invalid datatype");
    }
}
}
