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
#pragma once
#include "binary_writer.h"
#include <nncase/runtime/compiler_defs.h>
#include <nncase/runtime/nnil.h>

namespace nncase::codegen
{
class NNCASE_API nnil_builder
{
public:
    nnil_builder(binary_writer &writer)
        : writer_(writer) { }

    void emit_nop() { emit_opcode(runtime::nnil_nop); }
    void emit_dup() { emit_opcode(runtime::nnil_dup); }
    void emit_pop() { emit_opcode(runtime::nnil_pop); }
    void emit_lda_0() { emit_opcode(runtime::nnil_lda_0); }
    void emit_ldc_r4_0() { emit_opcode(runtime::nnil_ldc_r4_0); }
    void emit_ldc_r4_1() { emit_opcode(runtime::nnil_ldc_r4_1); }

    void emit_ldc_r4(float value)
    {
        emit_opcode(runtime::nnil_ldc_r4);
        writer_.write(runtime::nnil_ldc_r4_t { value });
    }

    void emit_abs() { emit_opcode(runtime::nnil_abs); }
    void emit_acos() { emit_opcode(runtime::nnil_acos); }
    void emit_asin() { emit_opcode(runtime::nnil_asin); }
    void emit_ceil() { emit_opcode(runtime::nnil_ceil); }
    void emit_cos() { emit_opcode(runtime::nnil_cos); }
    void emit_exp() { emit_opcode(runtime::nnil_exp); }
    void emit_floor() { emit_opcode(runtime::nnil_floor); }
    void emit_log() { emit_opcode(runtime::nnil_log); }
    void emit_neg() { emit_opcode(runtime::nnil_neg); }
    void emit_rsqrt() { emit_opcode(runtime::nnil_rsqrt); }
    void emit_sign() { emit_opcode(runtime::nnil_sign); }
    void emit_sin() { emit_opcode(runtime::nnil_sin); }
    void emit_sqrt() { emit_opcode(runtime::nnil_sqrt); }
    void emit_square() { emit_opcode(runtime::nnil_square); }
    void emit_tanh() { emit_opcode(runtime::nnil_tanh); }
    void emit_bitwise_not() { emit_opcode(runtime::nnil_bitwise_not); }
    void emit_logical_not() { emit_opcode(runtime::nnil_logical_not); }
    void emit_round() { emit_opcode(runtime::nnil_round); }
    void emit_add() { emit_opcode(runtime::nnil_add); }
    void emit_sub() { emit_opcode(runtime::nnil_sub); }
    void emit_mul() { emit_opcode(runtime::nnil_mul); }
    void emit_div() { emit_opcode(runtime::nnil_div); }
    void emit_min() { emit_opcode(runtime::nnil_min); }
    void emit_max() { emit_opcode(runtime::nnil_max); }
    void emit_pow() { emit_opcode(runtime::nnil_pow); }
    void emit_clamp() { emit_opcode(runtime::nnil_clamp); }

    // emit_erf
    void emit_erf() { emit_opcode(runtime::nnil_erf); }
    void emit_ret() { emit_opcode(runtime::nnil_ret); }

private:
    void emit_opcode(runtime::nnil_opcode_t opcode) { writer_.write((uint8_t)opcode); }

private:
    binary_writer &writer_;
};
}
