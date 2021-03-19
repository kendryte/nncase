/* Copyright 2020 Canaan Inc.
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
#include "graph.h"
#include "ir_types.h"
#include <filesystem>
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

inline std::string to_string(const padding &value)
{
    return "{" + std::to_string(value.before) + ", " + std::to_string(value.after) + "}";
}

inline std::string to_string(const quant_param_t &value)
{
    std::string ret = "{";
    for (size_t i = 0; i < value.zero_point.size(); i++)
    {
        std::string item = "{" + std::to_string(value.zero_point[i]) + "*" + std::to_string(value.scale[i]) + "}, ";
        ret += item;
    }
    ret += "}";
    return ret;
}

template <typename Tv, typename T>
static size_t index_of(const Tv &v, const T &e)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        if (&v[i] == &e)
        {
            return i;
        }
    }
    return SIZE_MAX;
}

namespace ir
{
    inline std::string to_string(const shape_t &shape)
    {
        std::string str { '[' };
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (i != 0)
            {
                str.append(",");
            }
            str.append(std::to_string(shape[i]));
        }

        str += ']';
        return str;
    }

    inline std::string to_string(const axis_t &axis)
    {
        std::string str { '[' };
        for (size_t i = 0; i < axis.size(); i++)
        {
            if (i != 0)
            {
                str.append(",");
            }
            str.append(std::to_string(axis[i]));
        }

        str += ']';
        return str;
    }

    NNCASE_API void dump_graph(const ir::graph &src_graph, const std::filesystem::path &dst_path);
}
}
