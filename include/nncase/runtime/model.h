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
#include "datatypes.h"

namespace nncase::runtime
{
typedef uint32_t model_target_t;

NNCASE_CONSTEXPR model_target_t MODEL_TARGET_CPU = 0;

struct model_header
{
    uint32_t identifier;
    uint32_t version;
    uint32_t checksum;
    uint32_t flags;
    model_target_t target;
    uint32_t memories;
    uint32_t sections;
    uint32_t inputs;
    uint32_t outputs;
};

struct memory_desc
{
    memory_location_t type;
    uint32_t size;
};

struct section_desc
{
    char name[8];
    uint32_t offset;
    uint32_t size_in_file;
    uint32_t size;
};

NNCASE_CONSTEXPR uint32_t MODEL_IDENTIFIER = 'KMDL';
NNCASE_CONSTEXPR uint32_t MODEL_VERSION = 5;
}
