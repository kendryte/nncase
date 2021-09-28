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
#include <nncase/targets/target.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;
using namespace nncase::runtime;

evaluator::evaluator(const schedule::model_schedule_result &sched)
    : model_eval_(sched)
{
}

void evaluator::enable_ptq(target &target, ir::calibrate_method calib_method)
{
    return model_eval_.enable_ptq(target, calib_method);
}

void evaluator::evaluate(bool before_quant, size_t stage, bool record_output_buffers)
{
    return model_eval_.evaluate(before_quant, stage, record_output_buffers);
}

quantizer *evaluator::quantizer(const module_type_t &module_type)
{
    return model_eval_.module(module_type).quantizer();
}

void evaluator::begin_collect_distribution()
{
    model_eval_.end_sample();
}

void evaluator::end_sample()
{
    model_eval_.end_sample();
}

void evaluator::end_collect_distribution(const std::function<void(size_t cnt, size_t total)> &progress)
{
    model_eval_.end_collect_distribution(progress);
}

evaluate_tensor evaluator::memory_at(const output_connector &conn)
{
    return model_eval_.memory_at(conn);
}

evaluate_tensor evaluator::memory_at(const input_connector &conn)
{
    return model_eval_.memory_at(conn);
}

evaluate_tensor evaluator::input_at(size_t index)
{
    return model_eval_.input_at(index);
}

evaluate_tensor evaluator::output_at(size_t index)
{
    return model_eval_.output_at(index);
}
