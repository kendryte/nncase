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
#include "evaluate_context.h"
#include "evaluate_types.h"

namespace nncase::ir
{
class NNCASE_API evaluator
{
public:
    evaluator(const schedule::model_schedule_result &sched);
    evaluator(evaluator &) = delete;
    evaluator(evaluator &&) = default;

    void enable_ptq(target &target, ir::calibrate_method calib_method);
    void evaluate();

    ir::quantizer *quantizer(const module_type_t &module_type);
    void begin_collect_distribution();
    void end_sample();
    void end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress);

    evaluate_tensor memory_at(const output_connector &conn);
    evaluate_tensor memory_at(const input_connector &conn);

    evaluate_tensor input_at(size_t index);
    evaluate_tensor output_at(size_t index);

private:
    model_evaluate_context model_eval_;
};

NNCASE_API void register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, function_evaluate_context &)> evaluator);
}
