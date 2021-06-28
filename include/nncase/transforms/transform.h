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
#include <filesystem>
#include <nncase/ir/graph.h>
#include <nncase/ir/quantizer.h>
#include <vector>

namespace nncase
{
class target;

namespace schedule
{
    struct schedule_context;
}

namespace ir
{
    class quantizer;

    namespace transforms
    {
        struct transform_context
        {
            transform_context(ir::graph &graph, nncase::target &target) noexcept
                : graph(graph), target(target)
            {
            }

            virtual ~transform_context() = default;

            ir::graph &graph;
            nncase::target &target;
            ir::quantizer *quantizer;
            schedule::schedule_context *schedule_context;
            std::optional<std::filesystem::path> dump_dir;
            std::vector<node *> matched_nodes;
            std::vector<input_connector *> inputs;
            std::vector<output_connector *> outputs;
        };

        class NNCASE_API transform
        {
        public:
            transform(std::string name = "noname")
                : name_(name) { }
            virtual ~transform() = default;

            std::string name() { return name_; }

            virtual std::unique_ptr<transform_context> create_context(ir::graph &graph, nncase::target &target);
            bool try_match(node &node, transform_context &context);

            virtual void process(transform_context &context) = 0;

        protected:
            virtual bool skip_self_contained_check() const noexcept;
            virtual bool on_try_match(node &node, transform_context &context) = 0;
            std::string name_;
        };

        NNCASE_API void link(ir::output_connector &old_c, ir::output_connector &new_c, ir::quantizer *quantizer = nullptr);
    }
}
}
