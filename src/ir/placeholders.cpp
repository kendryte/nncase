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
#include <nncase/ir/placeholders.h>

using namespace nncase;
using namespace nncase::ir;

// Workaround for error LNK2019 unresolved external symbol "__declspec(dllimport) const nncase::ir::ignore_node::`vftable'"
#ifdef _MSC_VER
static ignore_node dummy(dt_float32, shape_t { 1 });
#endif
