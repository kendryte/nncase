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
#include <nncase/ir/ops/compare.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(EQUAL)
{
    convert_compare(op, compare_equal);
}

DEFINE_TFLITE_LOWER(NOT_EQUAL)
{
    convert_compare(op, compare_not_equal);
}

DEFINE_TFLITE_LOWER(GREATER)
{
    convert_compare(op, compare_greater);
}

DEFINE_TFLITE_LOWER(GREATER_EQUAL)
{
    convert_compare(op, compare_greater_equal);
}

DEFINE_TFLITE_LOWER(LESS)
{
    convert_compare(op, compare_less);
}

DEFINE_TFLITE_LOWER(LESS_EQUAL)
{
    convert_compare(op, compare_less_equal);
}

void tflite_importer::convert_compare(const tflite::Operator &op, compare_op_t compare_op)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto input_type = to_data_type(input_a.type());
    auto &input_b = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);

    auto cmp = graph_.emplace<compare>(compare_op, input_type, get_shape(input_a.shape()), get_shape(input_b.shape()));
    cmp->name(std::string(output.name()->string_view()) + "/" + compare_op_to_string(compare_op));

    link_input_tensor(&cmp->input_a(), op.inputs()->Get(0));
    link_input_tensor(&cmp->input_b(), op.inputs()->Get(1));
    link_output_tensor(op.outputs()->Get(0), &cmp->output());
}
