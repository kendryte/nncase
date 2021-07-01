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
#pragma once
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/quantizer.h>
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/compiler_defs.h>
#include <nncase/schedule/scheduler.h>
#include <unordered_map>

namespace nncase::ir
{
class quantizer;

class NNCASE_API evaluate_tensor
{
public:
    evaluate_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> buffer);

    datatype_t datatype() const noexcept { return datatype_; }
    const runtime_shape_t &shape() const noexcept { return shape_; }
    const runtime_shape_t &strides() const noexcept { return strides_; }
    gsl::span<gsl::byte> buffer() const noexcept { return buffer_; }

private:
    datatype_t datatype_;
    runtime_shape_t shape_;
    runtime_shape_t strides_;
    gsl::span<gsl::byte> buffer_;
};

class NNCASE_API module_evaluate_context
{
public:
    module_evaluate_context(const schedule::module_schedule_result &sched);
    module_evaluate_context(module_evaluate_context &) = delete;
    module_evaluate_context(module_evaluate_context &&) = default;

    evaluate_tensor memory_at(const output_connector &conn);

    evaluate_tensor memory_at(const input_connector &conn)
    {
        return memory_at(*conn.connection());
    }

    evaluate_tensor input_at(size_t index)
    {
        return memory_at(*inputs_[index]);
    }

    evaluate_tensor output_at(size_t index)
    {
        return memory_at(*outputs_[index]);
    }

    ir::quantizer *quantizer() noexcept { return quantizer_.get(); }

    void enable_ptq(target &target, ir::calibrate_method calib_method);
    void evaluate();

    void begin_collect_distribution();
    void end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress);

private:
    const schedule::module_schedule_result &sched_;
    std::unordered_map<memory_location_t, std::unique_ptr<std::byte[]>> memory_pools_;

    std::vector<output_connector *> inputs_;
    std::vector<input_connector *> outputs_;
    std::unique_ptr<ir::quantizer> quantizer_;
};

class NNCASE_API evaluator
{
public:
    evaluator(const schedule::schedule_result &sched);
    evaluator(evaluator &) = delete;
    evaluator(evaluator &&) = default;

    module_evaluate_context &module_context(ir::graph &graph);
    module_evaluate_context &main_module_context();

    void enable_ptq(target &target, ir::calibrate_method calib_method);
    void evaluate();

    void begin_collect_distribution();
    void end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress);

    evaluate_tensor memory_at(const output_connector &conn);

    evaluate_tensor memory_at(const input_connector &conn)
    {
        return memory_at(*conn.connection());
    }

    evaluate_tensor input_at(size_t index)
    {
        return main_module_context().input_at(index);
    }

    evaluate_tensor output_at(size_t index)
    {
        return main_module_context().output_at(index);
    }

private:
    const schedule::schedule_result &sched_;
    std::unordered_map<ir::graph *, module_evaluate_context> module_ctxs_;
};

NNCASE_API void register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, module_evaluate_context &)> evaluator);
}
