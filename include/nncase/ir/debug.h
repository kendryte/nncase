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

    class dump_to_gml
    {

    public:
        dump_to_gml(const ir::graph &src_graph, std::filesystem::path dst_path);
        ~dump_to_gml();

        size_t get_id_from_node(const node *n);
        void step(const node &n, size_t force_gid = SIZE_MAX);
        void save();

    private:
        const ir::graph &src_graph;
        std::map<const node *, size_t> node_to_id;
        void *gml;
        std::filesystem::path dst_path;
        size_t next_id;
    };
}
}
