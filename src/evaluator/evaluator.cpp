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
#include "nncase/ir/quantizer.h"
#include <chrono>
#include <nncase/ir/evaluator.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/runtime_type_utils.h>
#include <nncase/targets/target.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
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

evaluate_tensor::evaluate_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> buffer)
    : datatype_(datatype), shape_(std::move(shape)), strides_(std::move(strides)), buffer_(buffer)
{
}

module_evaluate_context::module_evaluate_context(const module_schedule_result &sched)
    : sched_(sched), quantizer_(nullptr)
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
            auto dest = memory_at(rnode.output()).buffer().as_span<std::byte>();
            std::copy(std::begin(src), std::end(src), dest.begin());
        }
    }
}

evaluate_tensor module_evaluate_context::memory_at(const output_connector &conn)
{
    auto &alloc = sched_.allocations.at(&conn);
    auto &memory_pool = memory_pools_.at(alloc.memory_location);
    gsl::span<gsl::byte> buffer(reinterpret_cast<gsl::byte *>(memory_pool.get() + alloc.start), alloc.size);
    return evaluate_tensor(alloc.type, to(alloc.shape), to(alloc.strides), buffer);
}

void module_evaluate_context::enable_ptq(target &target, ir::calibrate_method calib_method)
{
    quantizer_ = target.create_quantizer(sched_.graph->module_type(), calib_method);
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

        if (quantizer_)
        {
            for (auto out : node->outputs())
            {
                if (out->attributes() & cnctr_attr_need_quantize)
                {
                    if (!quantizer_->has_record(*out))
                    {
                        auto mem = memory_at(*out);
                        if (mem.datatype() == dt_bfloat16)
                        {
                            auto buffer = mem.buffer().as_span<bfloat16>();
                            quantizer_->record(*out, buffer);
                        }
                        else
                        {
                            auto buffer = mem.buffer().as_span<float>();
                            quantizer_->record(*out, buffer);
                        }
                    }
                }
            }
        }
#if PROFILE
        std::cout << node_opcode_names(node->runtime_opcode()) << ": " << duration.count() / 1e6 << "ms" << std::endl;
#endif
    }

    if (quantizer_)
        quantizer_->reset_record();

#if PROFILE
    std::cout << "Total: " << total_duration.count() / 1e6 << "ms" << std::endl;
#endif
}

void module_evaluate_context::begin_collect_distribution()
{
    if (quantizer_)
        quantizer_->begin_collect_distribution();
}

void module_evaluate_context::end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress)
{
    if (quantizer_)
        quantizer_->end_collect_distribution(progress);
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

evaluate_tensor evaluator::memory_at(const output_connector &conn)
{
    return main_module_context().memory_at(conn);
}

void evaluator::enable_ptq(target &target, ir::calibrate_method calib_method)
{
    for (auto &module_p : module_ctxs_)
        module_p.second.enable_ptq(target, calib_method);
}

void evaluator::evaluate()
{
    module_ctxs_.at(sched_.main_module).evaluate();
}

void evaluator::begin_collect_distribution()
{
    for (auto &module_p : module_ctxs_)
        module_p.second.begin_collect_distribution();
}

void evaluator::end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress)
{
    for (auto &module_p : module_ctxs_)
        module_p.second.end_collect_distribution(progress);
}
