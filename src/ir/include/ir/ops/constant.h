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
#include "../node.h"
#include <runtime/runtime_op_utility.h>
#include <vector>

namespace nncase
{
namespace ir
{
    class constant : public node
    {
    public:
        DEFINE_NODE_OPCODE(op_constant);

        output_connector &output() { return output_at(0); }

        xtl::span<const uint8_t> data() const noexcept { return data_; }

        template <class TShape, class... TDataArgs>
        constant(datatype_t type, TShape &&shape, TDataArgs... data_args)
            : data_(std::forward<TDataArgs>(data_args)...)
        {
            add_output("output", type, std::forward<TShape>(shape), mem_const);
        }

        template <class TShape>
        constant(datatype_t type, TShape &&shape, xtl::span<const uint8_t> data)
            : data_(data.begin(), data.end())
        {
            add_output("output", type, std::forward<TShape>(shape), mem_const);
        }

        template <class TScalar>
        constant(TScalar scalar)
            : data_(reinterpret_cast<const uint8_t *>(&scalar), reinterpret_cast<const uint8_t *>(&scalar) + sizeof(scalar))
        {
            add_output("output", runtime::to_datatype_v<TScalar>, shape_t { 1 }, mem_const);
        }

    private:
        std::vector<uint8_t> data_;
    };
}
}
