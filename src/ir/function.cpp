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
#include <nncase/ir/function.h>

using namespace nncase;
using namespace nncase::ir;

namespace {
size_t global_func_index = 0;
}

function_node::function_node(std::string name, std::vector<var> parameters,
                             expr body)
    : name_(std::move(name)),
      parameters_(std::move(parameters)),
      body_(std::move(body)) {}

function::function(std::vector<var> parameters, expr body)
    : function("func_" + std::to_string(global_func_index++),
               std::move(parameters), std::move(body)) {}

function::function(std::string name, std::vector<var> parameters, expr body)
    : expr_t(std::in_place, std::move(name), std::move(parameters),
             std::move(body)) {}
