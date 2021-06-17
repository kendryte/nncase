/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SQUARED_DIFFERENCE)
{
    auto &input_x = get_tensor(op.inputs(), 0);
    auto &input_y = get_tensor(op.inputs(), 1);

    auto in_shape_x = get_shape(input_x.shape());
    auto in_shape_y = get_shape(input_y.shape());

    auto sub = graph_.emplace<binary>(binary_sub, in_shape_x, in_shape_y, value_range<float>::full());
    auto mul = graph_.emplace<unary>(unary_square, sub->output().shape());

    sub->name(get_tensor(op.outputs(), 0).name()->string_view());
    mul->name(get_tensor(op.outputs(), 0).name()->string_view());

    mul->input().connect(sub->output());

    link_input_tensor(&sub->input_a(), op.inputs()->Get(0));
    link_input_tensor(&sub->input_b(), op.inputs()->Get(1));
    link_output_tensor(op.outputs()->Get(0), &mul->output());
}
