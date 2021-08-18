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
#include <chrono>
#include <nncase/ir/evaluator.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/quantizer.h>
#include <nncase/ir/runtime_type_utils.h>
#include <nncase/targets/target.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;
namespace chrono = std::chrono;

namespace
{
std::unordered_map<node_opcode, std::function<void(ir::node &, function_evaluate_context &)>> g_evaluators;

auto &get_evaluator(node_opcode opcode)
{
    auto it = g_evaluators.find(opcode);
    if (it == std::end(g_evaluators))
        throw std::runtime_error("Evaluator for " + std::string(opcode.name) + " is not found");
    return it->second;
}
}

void nncase::ir::register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, function_evaluate_context &)> evaluator)
{
    g_evaluators.emplace(opcode, std::move(evaluator));
}

evaluate_tensor::evaluate_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> buffer)
    : datatype_(datatype), shape_(std::move(shape)), strides_(std::move(strides)), buffer_(buffer)
{
}

function_evaluate_context::function_evaluate_context(const function_schedule_result &sched, module_evaluate_context &mod_eval)
    : sched_(sched), mod_eval_(mod_eval)
{
    input_pool_ = std::make_unique<std::byte[]>(sched.input_pool_size);
    output_pool_ = std::make_unique<std::byte[]>(sched.output_pool_size);

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

evaluate_tensor function_evaluate_context::memory_at(const output_connector &conn)
{
    auto &alloc = module().sched().allocations.at(&conn);
    std::byte *base;
    switch (alloc.memory_location)
    {
    case mem_input:
        base = input_pool_.get();
        break;
    case mem_output:
        base = output_pool_.get();
        break;
    default:
        base = module().memory_pool(alloc.memory_location);
        break;
    }

    gsl::span<gsl::byte> buffer(reinterpret_cast<gsl::byte *>(base + alloc.start), alloc.size);
    return evaluate_tensor(alloc.type, to(alloc.shape), to(alloc.strides), buffer);
}

void function_evaluate_context::evaluate()
{
    using clock = chrono::high_resolution_clock;
    chrono::nanoseconds total_duration = {};
    auto quantizer = module().quantizer();

    for (auto &&node : sched_.compute_sequence)
    {
        auto &evaluator = get_evaluator(node->runtime_opcode());

        auto start = clock::now();
        evaluator(*node, *this);
        auto duration = clock::now() - start;
        total_duration += duration;

        if (quantizer)
        {
            for (auto out : node->outputs())
            {
                if (out->attributes() & cnctr_attr_need_quantize)
                {
                    if (!quantizer->has_record(*out))
                    {
                        auto mem = memory_at(*out);
                        auto dtype = mem.datatype();
                        if (dtype == dt_bfloat16)
                        {
                            auto buffer = mem.buffer().as_span<bfloat16>();
                            quantizer->record(*out, buffer);
                        }
                        else if (dtype == dt_float32)
                        {
                            auto buffer = mem.buffer().as_span<float>();
                            quantizer->record(*out, buffer);
                        }
                        else
                        {
                            throw std::runtime_error("Quantizer doesn't support datatype of " + std::string(datatype_names(dtype)));
                        }
                    }
                }
            }
        }
    }
}

module_evaluate_context::module_evaluate_context(const module_schedule_result &sched, model_evaluate_context &model_eval)
    : sched_(sched), model_eval_(model_eval), quantizer_(nullptr)
{
    for (auto &&usage : sched.max_usages)
        memory_pools_.emplace(usage.first, std::make_unique<std::byte[]>(usage.second));

    for (auto &func : sched.functions)
    {
        functions_.emplace(std::piecewise_construct,
            std::forward_as_tuple(func.graph),
            std::forward_as_tuple(func, *this));
    }
}

std::byte *module_evaluate_context::memory_pool(memory_location_t location) const
{
    return memory_pools_.at(location).get();
}

function_evaluate_context &module_evaluate_context::function(ir::graph &function)
{
    return functions_.at(&function);
}

void module_evaluate_context::enable_ptq(target &target, ir::calibrate_method calib_method)
{
    quantizer_ = target.create_quantizer(sched_.type, calib_method);
}

void module_evaluate_context::begin_collect_distribution()
{
    if (quantizer_)
        quantizer_->begin_collect_distribution();
}

void module_evaluate_context::end_sample()
{
    if (quantizer_)
        quantizer_->end_sample();
}

void module_evaluate_context::end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress)
{
    if (quantizer_)
        quantizer_->end_collect_distribution(progress);
}

model_evaluate_context::model_evaluate_context(const schedule::model_schedule_result &sched)
    : sched_(sched)
{
    for (auto &module : sched.modules)
        module_ctxs_.emplace(std::piecewise_construct, std::forward_as_tuple(module.type), std::forward_as_tuple(module, *this));
}

function_evaluate_context &model_evaluate_context::entrypoint()
{
    auto func = sched_.entry_function;
    return module_ctxs_.at(func->module->type).function(*func->graph);
}

module_evaluate_context &model_evaluate_context::module(const module_type_t &module_type)
{
    return module_ctxs_.at(module_type);
}

void model_evaluate_context::enable_ptq(target &target, ir::calibrate_method calib_method)
{
    for (auto &mod : module_ctxs_)
        mod.second.enable_ptq(target, calib_method);
}

void model_evaluate_context::begin_collect_distribution()
{
    for (auto &mod : module_ctxs_)
        mod.second.begin_collect_distribution();
}

void model_evaluate_context::end_sample()
{
    for (auto &mod : module_ctxs_)
        mod.second.end_sample();
}

void model_evaluate_context::end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress)
{
    for (auto &mod : module_ctxs_)
        mod.second.end_collect_distribution(progress);
}

void model_evaluate_context::evaluate()
{
    entrypoint().evaluate();
}
