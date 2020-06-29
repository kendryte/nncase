/* Copyright 2019-2020 Canaan Inc.
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
#include "../datatypes.h"
#include "binary_writer.h"
#include "span_reader.h"

namespace nncase
{
namespace runtime
{
    typedef enum _nnil_opcode
    {
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
        nnil_square = 0x29,
        nnil_add = 0x40,
        nnil_sub = 0x41,
        nnil_mul = 0x42,
        nnil_div = 0x43,
        nnil_min = 0x44,
        nnil_max = 0x45,
        nnil_clamp = 0x80,
        nnil_ret = 0xA0
    } nnil_opcode_t;

    typedef struct _nnil_ldc_r4
    {
        float r4;
    } nnil_ldc_r4_t;

    typedef struct _nnil_op
    {
        nnil_opcode_t opcode;

        union {
            nnil_ldc_r4_t ldc_r4;
        };
    } nnil_op_t;

    class nnil_builder
    {
    public:
        nnil_builder(binary_writer &writer)
            : writer_(writer) {}

        void emit_nop() { emit_opcode(nnil_nop); }
        void emit_dup() { emit_opcode(nnil_dup); }
        void emit_pop() { emit_opcode(nnil_pop); }
        void emit_lda_0() { emit_opcode(nnil_lda_0); }
        void emit_ldc_r4_0() { emit_opcode(nnil_ldc_r4_0); }
        void emit_ldc_r4_1() { emit_opcode(nnil_ldc_r4_1); }

        void emit_ldc_r4(float value)
        {
            emit_opcode(nnil_ldc_r4);
            writer_.write(nnil_ldc_r4_t { value });
        }

        void emit_abs() { emit_opcode(nnil_abs); }
        void emit_ceil() { emit_opcode(nnil_ceil); }
        void emit_cos() { emit_opcode(nnil_cos); }
        void emit_exp() { emit_opcode(nnil_exp); }
        void emit_floor() { emit_opcode(nnil_floor); }
        void emit_log() { emit_opcode(nnil_log); }
        void emit_neg() { emit_opcode(nnil_neg); }
        void emit_rsqrt() { emit_opcode(nnil_rsqrt); }
        void emit_sin() { emit_opcode(nnil_sin); }
        void emit_square() { emit_opcode(nnil_square); }
        void emit_add() { emit_opcode(nnil_add); }
        void emit_sub() { emit_opcode(nnil_sub); }
        void emit_mul() { emit_opcode(nnil_mul); }
        void emit_div() { emit_opcode(nnil_div); }
        void emit_min() { emit_opcode(nnil_min); }
        void emit_max() { emit_opcode(nnil_max); }
        void emit_clamp() { emit_opcode(nnil_clamp); }

        void emit_ret() { emit_opcode(nnil_ret); }

    private:
        void emit_opcode(nnil_opcode_t opcode) { writer_.write((uint8_t)opcode); }

    private:
        binary_writer &writer_;
    };

    class nnil_reader
    {
    public:
        nnil_reader(span_reader &reader)
            : reader_(reader) {}

        bool avail() const noexcept { return !reader_.empty(); }

        nnil_op_t next()
        {
            assert(avail());
            nnil_op_t op;
            op.opcode = (nnil_opcode_t)reader_.read<uint8_t>();

            switch (op.opcode)
            {
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

    class nnil_evalstack
    {
    public:
        nnil_evalstack() noexcept
            : top(0)
        {
        }

        void push(float value)
        {
            if (top < _stack.size())
                _stack[top++] = value;
            else
                NNCASE_THROW(std::runtime_error, "stack overflow");
        }

        float pop()
        {
            if (top > 0)
                return _stack[--top];
            else
                NNCASE_THROW(std::runtime_error, "stack underflow");
        }

        void dup()
        {
            if (top > 0)
            {
                _stack[top] = _stack[top - 1];
                top++;
            }
            else
            {
                NNCASE_THROW(std::runtime_error, "empty stack");
            }
        }

    private:
        std::array<float, 64> _stack;
        size_t top;
    };
}
}
