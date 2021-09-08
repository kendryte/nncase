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
#include <nncase/ir/constant.h>
#include <nncase/ir/math/functional.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

#define DEFINE_UNARY_TFLITE_LOWER(tflite_name, unary_op)                       \
    DEFINE_TFLITE_LOWER(tflite_name) { convert_unary(op, unary_op); }

DEFINE_UNARY_TFLITE_LOWER(ABS, unary_abs)
DEFINE_UNARY_TFLITE_LOWER(CEIL, unary_ceil)
DEFINE_UNARY_TFLITE_LOWER(COS, unary_cos)
DEFINE_UNARY_TFLITE_LOWER(EXP, unary_exp)
DEFINE_UNARY_TFLITE_LOWER(FLOOR, unary_floor)
DEFINE_UNARY_TFLITE_LOWER(LOG, unary_log)
DEFINE_UNARY_TFLITE_LOWER(NEG, unary_neg)
DEFINE_UNARY_TFLITE_LOWER(ROUND, unary_round)
DEFINE_UNARY_TFLITE_LOWER(RSQRT, unary_rsqrt)
DEFINE_UNARY_TFLITE_LOWER(SIN, unary_sin)
DEFINE_UNARY_TFLITE_LOWER(SQRT, unary_sqrt)
DEFINE_UNARY_TFLITE_LOWER(SQUARE, unary_square)
DEFINE_UNARY_TFLITE_LOWER(TANH, unary_tanh)

void tflite_importer::convert_unary(const tflite::Operator &op,
                                    unary_op_t unary_op) {
    auto [input] = get_input_exprs(op, 0);
    auto node = F::unary(unary_op, input);
    set_output_expr(op, 0, node);
}
