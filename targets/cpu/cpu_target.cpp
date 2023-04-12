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
#include "cpu_target.h"
#include <nncase/plugin_loader.h>

#if defined(_MSC_VER)
#define CPU_TARGET_API __declspec(dllexport)
#else
#define CPU_TARGET_API __attribute__((visibility("default")))
#endif

using namespace nncase;
using namespace nncase::targets;
using namespace nncase::runtime;

extern "C" {
CPU_TARGET_API target *create_target() { return new cpu_target(); }
}
