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
#include <chrono>
#include <nncase/ir/evaluator.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
namespace chrono = std::chrono;

#define PROFILE 0

namespace
{
std::unordered_map<node_opcode, std::function<void(ir::node &, module_evaluate_context &)>> g_evaluators;

auto &get_evaluator(node_opcode opcode)
{
    auto it = g_evaluators.find(opcode);
    if (it == std::end(g_evaluators))
        throw std::runtime_error("Evaluator for " + std::string(opcode.name) + " is not found");
    return it->second;
}
}

void nncase::ir::register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, module_evaluate_context &)> evaluator)
{
    g_evaluators.emplace(opcode, std::move(evaluator));
}

module_evaluate_context::module_evaluate_context(const module_schedule_result &sched)
    : sched_(sched)
{
    for (auto &&usage : sched.max_usages)
        memory_pools_.emplace(usage.first, std::make_unique<std::byte[]>(usage.second));

    for (auto &&node : sched.compute_sequence)
    {
        auto &opcode = node->runtime_opcode();
        if (opcode == op_input_node)
        {
            inputs_.emplace_back(&node->output_at(0));
        }
        else if (opcode == op_output_node)
        {
            outputs_.emplace_back(&node->input_at(0));
        }
        else if (opcode == op_constant)
        {
            auto &rnode = static_cast<constant &>(*node);
            auto src = rnode.data();
            auto dest = memory_at(rnode.output()).view();
            std::copy(std::begin(src), std::end(src), dest.begin());
        }
    }
}

eval_result<> module_evaluate_context::memory_at(const output_connector &conn)
{
    auto &alloc = sched_.allocations.at(&conn);
    auto &memory_pool = memory_pools_.at(alloc.memory_location);
    return {
        { memory_pool.get() + alloc.start, alloc.size },
        runtime::convert_shape_type(alloc.shape, alloc.type, dt_uint8),
        runtime::convert_strides_type(alloc.strides, alloc.type, dt_uint8)
    };
}

void module_evaluate_context::evaluate()
{
    using clock = chrono::high_resolution_clock;
    chrono::nanoseconds total_duration = {};

    for (auto &&node : sched_.compute_sequence)
    {
        auto &evaluator = get_evaluator(node->runtime_opcode());

        auto start = clock::now();
        evaluator(*node, *this);
        auto duration = clock::now() - start;
        total_duration += duration;
#if PROFILE
        std::cout << node_opcode_names(node->runtime_opcode()) << ": " << duration.count() / 1e6 << "ms" << std::endl;
#endif
    }

#if PROFILE
    std::cout << "Total: " << total_duration.count() / 1e6 << "ms" << std::endl;
#endif
}

evaluator::evaluator(const schedule::schedule_result &sched)
    : sched_(sched)
{
    for (auto &module_p : sched.modules)
        module_ctxs_.emplace(module_p.first, module_p.second);
}

module_evaluate_context &evaluator::module_context(ir::graph &graph)
{
    return module_ctxs_.at(&graph);
}

module_evaluate_context &evaluator::main_module_context()
{
    return module_context(*sched_.main_module);
}

eval_result<> evaluator::memory_at(const output_connector &conn)
{
    return main_module_context().memory_at(conn);
}

void evaluator::evaluate()
{
    module_ctxs_.at(sched_.main_module).evaluate();
}
