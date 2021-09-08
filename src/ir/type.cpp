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
#include <nncase/ir/type.h>

using namespace nncase;
using namespace nncase::ir;

tensor_type_node::tensor_type_node(datatype_t elem_type, shape_t shape)
    : elem_type_(elem_type), shape_(std::move(shape)) {}

tensor_type::tensor_type(datatype_t elem_type, shape_t shape)
    : object_t(std::in_place, elem_type, std::move(shape)) {}
