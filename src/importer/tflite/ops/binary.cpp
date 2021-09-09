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
#include <nncase/ir/math/functional.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ADD) {
    convert_binary(
        op, binary_add,
        op.builtin_options_as_AddOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(DIV) {
    convert_binary(
        op, binary_div,
        op.builtin_options_as_DivOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(FLOOR_DIV) {
    auto [lhs, rhs] = get_input_exprs(op, 0, 1);
    auto node = F::floor(lhs / rhs);
    set_output_expr(op, 0, node);
}

DEFINE_TFLITE_LOWER(FLOOR_MOD) { convert_binary(op, binary_mod); }
DEFINE_TFLITE_LOWER(MAXIMUM) { convert_binary(op, binary_max); }
DEFINE_TFLITE_LOWER(MINIMUM) { convert_binary(op, binary_min); }

DEFINE_TFLITE_LOWER(MUL) {
    convert_binary(
        op, binary_mul,
        op.builtin_options_as_MulOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(POW) { convert_binary(op, binary_pow); }

DEFINE_TFLITE_LOWER(SUB) {
    convert_binary(
        op, binary_sub,
        op.builtin_options_as_SubOptions()->fused_activation_function());
}

void tflite_importer::convert_binary(
    const tflite::Operator &op, binary_op_t binary_op,
    tflite::ActivationFunctionType activation) {
    auto [lhs, rhs] = get_input_exprs(op, 0, 1);
    auto node = F::binary(binary_op, lhs, rhs);
    if (activation != tflite::ActivationFunctionType_NONE)
        node = activate(node, activation);
    set_output_expr(op, 0, node);
}
