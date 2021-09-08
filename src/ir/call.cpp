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
#include <nncase/ir/call.h>

using namespace nncase;
using namespace nncase::ir;

call_node::call_node(expr target, std::vector<expr> arguments)
    : target_(std::move(target)), arguments_(std::move(arguments)) {
    if (!target_.is_a<function>() && !target_.is_a<op>()) {
        throw std::invalid_argument(
            "Call: target should be either a function or an op.");
    }
}

call::call(expr target, std::vector<expr> arguments)
    : object_t(std::in_place, std::move(target), std::move(arguments)) {}
