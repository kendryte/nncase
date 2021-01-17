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
#include "../graph.h"
#include <xtensor/xtensor.hpp>

namespace nncase::ir
{
class call : public node
{
public:
    DEFINE_NODE_OPCODE(op_call);

    graph &target() const noexcept { return target_; }

    call(graph &target);

    input_connector &outer_connector(input_node &target_input);
    input_connector &outer_connector(input_connector &target_input);
    output_connector &outer_connector(output_node &target_output);
    output_connector &outer_connector(output_connector &target_output);

protected:
    bool properties_equal(node &other) const override;

private:
    graph &target_;
};
}
