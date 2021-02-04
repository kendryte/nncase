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
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/schedule/scheduler.h>
#include <unordered_map>
#include <xtensor/xadapt.hpp>

namespace nncase::ir
{
class NNCASE_API module_evaluate_context
{
public:
    module_evaluate_context(const schedule::module_schedule_result &sched);
    module_evaluate_context(module_evaluate_context &) = delete;
    module_evaluate_context(module_evaluate_context &&) = default;

    runtime::runtime_tensor memory_at(const output_connector &conn);

    runtime::runtime_tensor memory_at(const input_connector &conn)
    {
        return memory_at(*conn.connection());
    }

    runtime::runtime_tensor input_at(size_t index)
    {
        return memory_at(*inputs_[index]);
    }

    runtime::runtime_tensor output_at(size_t index)
    {
        return memory_at(*outputs_[index]);
    }

    void evaluate();

private:
    const schedule::module_schedule_result &sched_;
    std::unordered_map<memory_location_t, std::unique_ptr<std::byte[]>> memory_pools_;

    std::vector<output_connector *> inputs_;
    std::vector<input_connector *> outputs_;
};

class NNCASE_API evaluator
{
public:
    evaluator(const schedule::schedule_result &sched);
    evaluator(evaluator &) = delete;

    module_evaluate_context &module_context(ir::graph &graph);
    module_evaluate_context &main_module_context();
    void evaluate();

    runtime::runtime_tensor memory_at(const output_connector &conn);

    runtime::runtime_tensor memory_at(const input_connector &conn)
    {
        return memory_at(*conn.connection());
    }

    runtime::runtime_tensor input_at(size_t index)
    {
        return main_module_context().input_at(index);
    }

    runtime::runtime_tensor output_at(size_t index)
    {
        return main_module_context().output_at(index);
    }

private:
    const schedule::schedule_result &sched_;
    std::unordered_map<ir::graph *, module_evaluate_context> module_ctxs_;
};

NNCASE_API void register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, module_evaluate_context &)> evaluator);
}
