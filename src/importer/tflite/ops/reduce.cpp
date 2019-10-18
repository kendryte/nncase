/* Copyright 2019 Canaan Inc.
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
#include "../tflite_importer.h"
#include <ir/ops/reduce.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(REDUCE_MAX)
{
    convert_reduce(op, reduce_max, std::numeric_limits<float>::lowest());
}

DEFINE_TFLITE_LOWER(MEAN)
{
    convert_reduce(op, reduce_mean, 0.f);
}

DEFINE_TFLITE_LOWER(REDUCE_MIN)
{
    convert_reduce(op, reduce_min, std::numeric_limits<float>::max());
}

DEFINE_TFLITE_LOWER(SUM)
{
    convert_reduce(op, reduce_sum, 0.f);
}

void tflite_importer::convert_reduce(const tflite::Operator &op, reduce_op_t reduce_op, float init_value)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto axis = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto &options = *op.builtin_options_as_ReducerOptions();

    auto node = graph_.emplace<reduce>(reduce_op, get_shape(input.shape()), std::move(axis), init_value, options.keep_dims());

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
