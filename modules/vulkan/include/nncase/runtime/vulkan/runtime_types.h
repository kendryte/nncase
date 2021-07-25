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
#include <nncase/runtime/datatypes.h>

BEGIN_NS_NNCASE_RT_MODULE(vulkan)

enum class opcode_t : uint8_t
{
    ldbuf,
    dupbuf,
    popbuf,
    ldpipeline,
    dispatch
};

enum class shader_type_t : uint8_t
{
    compute
};

struct ldbuf_op_t
{
    opcode_t opcode = opcode_t::ldbuf;
    uint8_t reserved0[3];

    memory_range memory;
};

struct dupbuf_op_t
{
    opcode_t opcode = opcode_t::dupbuf;
    uint8_t reserved0[3];
};

struct popbuf_op_t
{
    opcode_t opcode = opcode_t::popbuf;
    uint8_t reserved0[3];
};

struct ldpipeline_op_t
{
    opcode_t opcode = opcode_t::ldpipeline;
    uint8_t reserved0[1];

    shader_type_t shader_type;
    uint8_t buffers;
    uint32_t shader_start;
    uint32_t shader_size;
};

struct dispatch_op_t
{
    opcode_t opcode = opcode_t::dispatch;
    uint8_t reserved0[3];
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

END_NS_NNCASE_RT_MODULE
