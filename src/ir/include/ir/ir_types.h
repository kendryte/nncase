/* Copyright 2019 Canaan Inc.
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
#include <datatypes.h>
#include <xtensor/xshape.hpp>

namespace nncase
{
namespace ir
{
    using shape_t = xt::dynamic_shape<std::size_t>;
    using axis_t = xt::dynamic_shape<int32_t>;

    enum node_attributes
    {
        node_attr_none = 0,
        node_attr_action = 1
    };

    inline std::string to_string(const shape_t &shape)
    {
        std::string str;
        for (size_t i = 0; i < shape.size(); i++)
        {
            str.append(std::to_string(shape[i]));
            if (i != shape.size() - 1)
                str.append("x");
        }

        return str;
    }
}
}
