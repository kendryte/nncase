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
#pragma once
#include <nncase/ir/debug.h>
#include <nncase/ir/ir_types.h>
#include <nncase/ir/ops/convert.h>

namespace nncase::importer
{
using ir::shape_t;

template <class Id>
void link_input_tensor_by_id(std::unordered_map<ir::input_connector *, Id> &input_tensors, ir::input_connector *conn, const Id &in_id,
    datatype_t in_type, const shape_t &in_shape, const std::string &tensor_name)
{
    input_tensors.emplace(conn, in_id);
    if (in_type != conn->type())
    {
        throw std::runtime_error(
            "Type must be same: \n"
            + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
            + tensor_name + "[input]"
            + "\n has type mismatch: \n["
            + std::string(datatype_names(conn->type())) + "] != ["
            + std::string(datatype_names(in_type)) + "]");
    }
    if (in_shape != conn->shape())
    {
        throw std::runtime_error(
            "Shape must be same: \n"
            + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
            + tensor_name + "[output]"
            + "\n has shape mismatch: \n"
            + ir::to_string(conn->shape()) + " != "
            + ir::to_string(in_shape) + "");
    }
}

template <class Id>
void link_output_tensor_by_id(std::unordered_map<Id, ir::output_connector *> &output_tensors, const Id &out_id, ir::output_connector *conn,
    datatype_t out_type, const shape_t &out_shape, const std::string &tensor_name)
{
    output_tensors.emplace(out_id, conn);
    if (out_type != conn->type())
    {
        throw std::runtime_error(
            "Type must be same: \n"
            + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
            + tensor_name + "[output]"
            + "\n has type mismatch: \n["
            + std::string(datatype_names(conn->type())) + "] != ["
            + std::string(datatype_names(out_type)) + "]");
    }

    if (out_shape != conn->shape())
    {
        throw std::runtime_error(
            "Shape must be same: \n"
            + conn->owner().name() + "[" + std::string(conn->owner().runtime_opcode().name) + "] != "
            + tensor_name + "[output]"
            + "\n has shape mismatch: \n"
            + ir::to_string(conn->shape()) + " != "
            + ir::to_string(out_shape) + "");
    }
}

template <class T = int32_t>
T get_positive(int32_t v, size_t length)
{
    return static_cast<T>(v < 0 ? v + length : v);
}

// place new node before exist node
// Node output -> NextNode input
template <class Node, class... Args>
Node *add_prev_node(ir::graph &graph, ir::input_connector &next_input, Args &&...args)
{
    auto node = graph.emplace<Node>(std::forward<Args>(args)...);
    next_input.connect(node->output());
    return node;
}

template <class Node, class... Args>
Node *add_next_node(ir::graph &graph, ir::output_connector &prev_output, Args &&...args)
{
    auto node = graph.emplace<Node>(std::forward<Args>(args)...);
    node->input().connect(prev_output);
    return node;
}
}