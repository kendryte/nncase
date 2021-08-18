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
#include "evaluate_types.h"
#include "quantizer.h"
#include <nncase/schedule/schedule_types.h>

namespace nncase
{
class target;
}

namespace nncase::ir
{
class module_evaluate_context;
class model_evaluate_context;

class NNCASE_API function_evaluate_context
{
public:
    function_evaluate_context(const schedule::function_schedule_result &sched, module_evaluate_context &mod_eval);
    function_evaluate_context(const function_evaluate_context &) = delete;
    function_evaluate_context(function_evaluate_context &&) = default;

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

    module_evaluate_context &module() const noexcept { return mod_eval_; }

    void evaluate();

private:
    const schedule::function_schedule_result &sched_;
    module_evaluate_context &mod_eval_;
    std::unique_ptr<std::byte[]> input_pool_;
    std::unique_ptr<std::byte[]> output_pool_;

    std::vector<output_connector *> inputs_;
    std::vector<input_connector *> outputs_;
};

class NNCASE_API module_evaluate_context
{
public:
    module_evaluate_context(const schedule::module_schedule_result &sched, model_evaluate_context &model_eval);
    module_evaluate_context(module_evaluate_context &) = delete;
    module_evaluate_context(module_evaluate_context &&) = default;

    const schedule::module_schedule_result &sched() const noexcept { return sched_; }
    std::byte *memory_pool(memory_location_t location) const;
    ir::quantizer *quantizer() noexcept { return quantizer_.get(); }
    function_evaluate_context &function(ir::graph &function);
    model_evaluate_context &model() const noexcept { return model_eval_; }

    void enable_ptq(target &target, ir::calibrate_method calib_method);
    void begin_collect_distribution();
    void end_sample();
    void end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress);

private:
    const schedule::module_schedule_result &sched_;
    model_evaluate_context &model_eval_;
    std::unordered_map<memory_location_t, std::unique_ptr<std::byte[]>> memory_pools_;

    std::vector<output_connector *> inputs_;
    std::vector<input_connector *> outputs_;
    std::unique_ptr<ir::quantizer> quantizer_;
    std::unordered_map<ir::graph *, function_evaluate_context> functions_;
};

class NNCASE_API model_evaluate_context
{
public:
    model_evaluate_context(const schedule::model_schedule_result &sched);
    model_evaluate_context(const model_evaluate_context &) = delete;
    model_evaluate_context(model_evaluate_context &&) = default;

    function_evaluate_context &entrypoint();
    module_evaluate_context &module(const module_type_t &module_type);

    evaluate_tensor memory_at(const output_connector &conn)
    {
        return entrypoint().memory_at(conn);
    }

    evaluate_tensor memory_at(const input_connector &conn)
    {
        return memory_at(*conn.connection());
    }

    evaluate_tensor input_at(size_t index)
    {
        return entrypoint().input_at(index);
    }

    evaluate_tensor output_at(size_t index)
    {
        return entrypoint().output_at(index);
    }

    void enable_ptq(nncase::target &target, ir::calibrate_method calib_method);
    void begin_collect_distribution();
    void end_sample();
    void end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress);

    void evaluate();

private:
    const schedule::model_schedule_result &sched_;
    std::unordered_map<module_type_t, module_evaluate_context> module_ctxs_;
};
}
