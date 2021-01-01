/* Copyright 2020 Canaan Inc.
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
#include <nncase/codegen/codegen.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::codegen;

namespace nncase::codegen
{
void register_neutral_emitters()
{
    disable_emitter(op_input_node);
    disable_emitter(op_output_node);
    disable_emitter(op_constant);
    disable_emitter(op_ignore_node);
    disable_emitter(op_uninitialized);
}
}
