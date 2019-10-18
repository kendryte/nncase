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
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ABS)
{
    convert_unary(op, unary_abs);
}

DEFINE_TFLITE_LOWER(CEIL)
{
    convert_unary(op, unary_ceil);
}

DEFINE_TFLITE_LOWER(COS)
{
    convert_unary(op, unary_cos);
}

DEFINE_TFLITE_LOWER(EXP)
{
    convert_unary(op, unary_exp);
}

DEFINE_TFLITE_LOWER(FLOOR)
{
    convert_unary(op, unary_floor);
}

DEFINE_TFLITE_LOWER(LOG)
{
    convert_unary(op, unary_log);
}

DEFINE_TFLITE_LOWER(NEG)
{
    convert_unary(op, unary_neg);
}

DEFINE_TFLITE_LOWER(RSQRT)
{
    convert_unary(op, unary_rsqrt);
}

DEFINE_TFLITE_LOWER(SIN)
{
    convert_unary(op, unary_sin);
}

void tflite_importer::convert_unary(const tflite::Operator &op, unary_op_t unary_op)
{
    auto &input = get_tensor(op.inputs(), 0);

    auto node = graph_.emplace<unary>(unary_op, get_shape(input.shape()));

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
