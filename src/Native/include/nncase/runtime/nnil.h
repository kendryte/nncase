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
#include "compiler_defs.h"
#include "span_reader.h"
#include <array>
#include <cassert>

BEGIN_NS_NNCASE_RUNTIME

typedef enum _nnil_opcode {
    nnil_nop = 0x00,
    nnil_dup = 0x01,
    nnil_pop = 0x02,
    nnil_lda_0 = 0x03,
    nnil_ldc_r4_0 = 0x04,
    nnil_ldc_r4_1 = 0x05,
    nnil_ldc_r4 = 0x06,
    nnil_abs = 0x20,
    nnil_ceil = 0x21,
    nnil_cos = 0x22,
    nnil_exp = 0x23,
    nnil_floor = 0x24,
    nnil_log = 0x25,
    nnil_neg = 0x26,
    nnil_rsqrt = 0x27,
    nnil_sin = 0x28,
    nnil_sqrt = 0x29,
    nnil_square = 0x2A,
    nnil_tanh = 0x2B,
    nnil_bitwise_not = 0x2C,
    nnil_logical_not = 0x2D,
    nnil_round = 0x2E,
    nnil_acos = 0x2F,
    nnil_asin = 0x30,
    nnil_sign = 0x31,
    nnil_add = 0x40,
    nnil_sub = 0x41,
    nnil_mul = 0x42,
    nnil_div = 0x43,
    nnil_min = 0x44,
    nnil_max = 0x45,
    nnil_pow = 0x46,
    nnil_clamp = 0x80,
    nnil_ret = 0xA0
} nnil_opcode_t;

typedef struct _nnil_ldc_r4 {
    float r4;
} nnil_ldc_r4_t;

typedef struct _nnil_op {
    nnil_opcode_t opcode;

    union {
        nnil_ldc_r4_t ldc_r4;
    };
} nnil_op_t;

class nnil_reader {
  public:
    nnil_reader(span_reader &reader) : reader_(reader) {}

    bool avail() const noexcept { return !reader_.empty(); }

    nnil_op_t next() {
        assert(avail());
        nnil_op_t op;
        op.opcode = (nnil_opcode_t)reader_.read<uint8_t>();

        switch (op.opcode) {
        case nnil_ldc_r4:
            op.ldc_r4 = reader_.read_unaligned<nnil_ldc_r4_t>();
            break;
        default:
            break;
        }

        return op;
    }

  private:
    span_reader &reader_;
};

class nnil_evalstack {
  public:
    nnil_evalstack() noexcept : top(0) {}

    void push(float value) {
        assert(top < _stack.size());
        _stack[top++] = value;
    }

    float pop() {
        assert(top > 0);
        return _stack[--top];
    }

    void dup() {
        assert(top > 0);
        _stack[top] = _stack[top - 1];
        top++;
    }

  private:
    std::array<float, 64> _stack;
    size_t top;
};

END_NS_NNCASE_RUNTIME
