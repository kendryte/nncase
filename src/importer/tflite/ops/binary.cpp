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
#include <ir/ops/binary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ADD)
{
    convert_binary(op, binary_add, op.builtin_options_as_AddOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(DIV)
{
    convert_binary(op, binary_div, op.builtin_options_as_DivOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(MAXIMUM)
{
    convert_binary(op, binary_max, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MINIMUM)
{
    convert_binary(op, binary_min, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MUL)
{
    convert_binary(op, binary_mul, op.builtin_options_as_MulOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(SUB)
{
    convert_binary(op, binary_sub, op.builtin_options_as_SubOptions()->fused_activation_function());
}

void tflite_importer::convert_binary(const tflite::Operator &op, binary_op_t binary_op, tflite::ActivationFunctionType activation)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);

    auto add = graph_.emplace<binary>(binary_op, get_shape(input_a.shape()), get_shape(input_b.shape()), to_float_clamp_range(activation));

    input_tensors_.emplace(&add->input_a(), op.inputs()->Get(0));
    input_tensors_.emplace(&add->input_b(), op.inputs()->Get(1));
    output_tensors_.emplace(op.outputs()->Get(0), &add->output());
}
