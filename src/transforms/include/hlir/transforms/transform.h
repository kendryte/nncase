/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/graph.h>
#include <vector>

namespace nncase
{
class target;

namespace hlir
{
    class quantizer;

    namespace transforms
    {
        struct transform_context
        {
            hlir::graph &graph;
            nncase::target &target;
            std::vector<node *> matched_nodes;
            std::vector<input_connector *> inputs;
            std::vector<output_connector *> outputs;
        };

        class transform
        {
        public:
            virtual ~transform() = default;

            bool try_match(node &node, transform_context &context);

            virtual void process(transform_context &context) = 0;

        protected:
            virtual bool skip_self_contained_check() const noexcept;
            virtual bool on_try_match(node &node, transform_context &context) = 0;
        };

        std::vector<input_connector *> dup(xtl::span<hlir::input_connector *const> connections);
        void link(hlir::output_connector &old_c, hlir::output_connector &new_c, hlir::quantizer *quantizer = nullptr);
    }
}
}
