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
#include <ir/ops/constant.h>
#include <ir/ops/reduce.h>
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(RELU)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_LeakyReluOptions();
    auto in_shape = get_shape(input.shape());

    auto zero = graph_.emplace<constant>(0.f);
    auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &max->output());
}

DEFINE_TFLITE_LOWER(LEAKY_RELU)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_LeakyReluOptions();
    auto in_shape = get_shape(input.shape());

    auto alpha = graph_.emplace<constant>(options.alpha());
    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&mul->input_a(), op.inputs()->Get(0));
    input_tensors_.emplace(&max->input_a(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &max->output());
}
