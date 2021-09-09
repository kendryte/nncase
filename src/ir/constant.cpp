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
#include <nncase/ir/constant.h>

using namespace nncase;
using namespace nncase::ir;

constant_node::constant_node(type value_type, std::vector<std::byte> data)
    : value_type_(std::move(value_type)), data_(std::move(data)) {}

constant::constant(type value_type, std::vector<std::byte> data)
    : object_t(std::in_place, std::move(value_type), std::move(data)) {}
