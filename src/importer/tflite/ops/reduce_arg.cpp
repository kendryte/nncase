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
#include "../tflite_importer.h"
#include <nncase/ir/ops/reduce_arg.h>
#include <nncase/ir/ops/transpose.h>
#include <schema_generated.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ARG_MAX)
{
    auto &options = *op.builtin_options_as_ArgMaxOptions();
    convert_reduce_arg(op, reduce_arg_max, options.output_type());
}

DEFINE_TFLITE_LOWER(ARG_MIN)
{
    auto &options = *op.builtin_options_as_ArgMinOptions();
    convert_reduce_arg(op, reduce_arg_min, options.output_type());
}

void tflite_importer::convert_reduce_arg(const tflite::Operator &op, reduce_arg_op_t reduce_arg_op, const tflite::TensorType output_type)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto axes = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    assert(axes.size() == 1);
    assert(output_type == tflite::TensorType_INT32 || output_type == tflite::TensorType_INT64);

    auto node = graph_.emplace<reduce_arg>(reduce_arg_op, to_data_type(input.type()), get_shape(input.shape()),
        to_data_type(output_type), axes[0], false);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
