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
#include <nncase/ir/var.h>

using namespace nncase;
using namespace nncase::ir;

namespace {
size_t global_var_index = 0;
}

var_node::var_node(std::string name, type type_annotation) noexcept
    : name_(std::move(name)), type_annotation_(std::move(type_annotation)) {}

var::var(type type_annotation)
    : var("var_" + std::to_string(global_var_index++),
          std::move(type_annotation)) {}

var::var(std::string name, type type_annotation)
    : object_t(std::in_place, std::move(name), std::move(type_annotation)) {}
