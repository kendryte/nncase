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
#include <nncase/ir/expr.h>
#include <nncase/ir/node.h>
#include <nncase/ir/var.h>
#include <nncase/runtime/stackvm/runtime_module.h>

using namespace nncase;
using namespace nncase::ir;

node::node(std::string name)
    : name_(std::move(name)), module_type_(runtime::stackvm::stackvm_module_type)
{
}

node::~node()
{
}

bool node::equals(node &other) const
{
    if (runtime_opcode() == other.runtime_opcode()
        && attributes() == other.attributes())
    {
        if (inputs().size() == other.inputs().size())
        {
            for (size_t i = 0; i < inputs().size(); i++)
            {
                if (input_at(i).connection() != other.input_at(i).connection())
                    return false;
            }

            return properties_equal(other);
        }
    }

    return false;
}
