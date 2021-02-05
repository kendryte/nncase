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
#include <array>
#include <cstdint>

#ifdef __riscv64
#define NNCASE_TARGET_K210_SIMULATOR 0
#include <kpu.h>
#else
#define NNCASE_TARGET_K210_SIMULATOR 1
#endif

namespace nncase
{
namespace runtime
{
    namespace k210
    {
#if NNCASE_TARGET_K210_SIMULATOR
    }
}
}
