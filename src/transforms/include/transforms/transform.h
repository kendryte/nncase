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
#include <ir/graph.h>
#include <vector>

namespace nncase
{
class target;

namespace transforms
{
    struct transform_context
    {
        ir::graph &graph;
        nncase::target &target;
        std::vector<ir::node *> matched_nodes;
        std::vector<ir::input_connector *> inputs;
        std::vector<ir::output_connector *> outputs;
    };

    class transform
    {
    public:
        bool try_match(ir::node &node, transform_context &context);

        virtual void process(transform_context &context) = 0;

    protected:
        virtual bool skip_self_contained_check() const noexcept;
        virtual bool on_try_match(ir::node &node, transform_context &context) = 0;
    };

    void transform_graph(ir::graph &graph, nncase::target &target, xtl::span<transform *> transforms);
    std::vector<ir::input_connector *> dup(xtl::span<ir::input_connector *const> connections);
}
}
