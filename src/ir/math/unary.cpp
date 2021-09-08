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
#include "nncase/runtime/datatypes.h"
#include <nncase/ir/math/unary.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::math;

unary_node::unary_node(unary_op_t unary_op) : unary_op_(unary_op) {
    add_parameter("input");
}

unary::unary(unary_op_t unary_op) : object_t(std::in_place, unary_op) {}
