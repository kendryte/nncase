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

enum class opcode_t : uint8_t {
    ldbuf,
    ldbufcopy,
    copybuf,
    ldpipeline,
    dispatch,
    ldbufbarrier,
    barrier
};

enum class shader_type_t : uint8_t { compute };

struct ldbuf_op_t {
    opcode_t opcode = opcode_t::ldbuf;
    uint8_t reserved0[3];

    memory_range memory;
};

struct ldpipeline_op_t {
    opcode_t opcode = opcode_t::ldpipeline;
    uint8_t reserved0[1];

    shader_type_t shader_type;
    uint8_t buffers;
    uint32_t shader_start;
    uint32_t shader_size;
};

struct ldbufcopy_op_t {
    opcode_t opcode = opcode_t::ldbufcopy;
    uint8_t reserved0[3];

    uint32_t src;
    uint32_t dest;
    uint32_t size;
};

struct copybuf_op_t {
    opcode_t opcode = opcode_t::copybuf;
    uint8_t reserved0[3];

    uint32_t regions;
};

struct dispatch_op_t {
    opcode_t opcode = opcode_t::dispatch;
    uint8_t reserved0[3];
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct ldbufbarrier_op_t {
    opcode_t opcode = opcode_t::ldbufbarrier;
    uint8_t reserved0[3];

    memory_range memory;
    uint32_t src_access_mask;
    uint32_t dest_access_mask;
};

struct barrier_op_t {
    opcode_t opcode = opcode_t::barrier;
    uint8_t reserved0[3];

    uint32_t src_stage;
    uint32_t dest_stage;
    uint32_t dependency_flags;
    uint32_t memory_barriers;
    uint32_t buffer_barriers;
};

END_NS_NNCASE_RT_MODULE
