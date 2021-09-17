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

invalid_type_node::invalid_type_node() {}

invalid_type_node::invalid_type_node(std::string reason)
    : reason_(std::move(reason)) {}

invalid_type::invalid_type() {}

invalid_type::invalid_type(std::string reason)
    : object_t(std::in_place, std::move(reason)) {}

tensor_type_node::tensor_type_node(datatype_t dtype, shape_t shape)
    : dtype_(dtype), shape_(std::move(shape)) {}

tensor_type::tensor_type(datatype_t dtype, shape_t shape)
    : object_t(std::in_place, dtype, std::move(shape)) {}

tuple_type_node::tuple_type_node(itlib::small_vector<type> fields)
    : fields_(std::move(fields)) {}

tuple_type::tuple_type(itlib::small_vector<type> fields)
    : object_t(std::in_place, std::move(fields)) {}

callable_type_node::callable_type_node(itlib::small_vector<type> parameters,
                                       type return_type)
    : parameters_(std::move(parameters)),
      return_type_(std::move(return_type)) {}

callable_type::callable_type(itlib::small_vector<type> parameters,
                             type return_type)
    : object_t(std::in_place, std::move(parameters), std::move(return_type)) {}
