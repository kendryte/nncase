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
#include "expr.h"
#include "opcode.h"

namespace nncase::ir
{
class output_connector;

class NNCASE_API base_connector
{
public:
    base_connector(std::string name, datatype_t type, TShape &&shape)
        : owner_(owner), name_(std::forward<TName>(name)), type_(type), shape_(std::forward<TShape>(shape))
    {
    }

    base_connector(base_connector &) = delete;
    base_connector(base_connector &&) = default;

    node &owner() const noexcept { return owner_; }
    const std::string &name() const noexcept { return name_; }
    datatype_t type() const noexcept { return type_; }
    const shape_t &shape() const noexcept { return shape_; }
    connector_attributes attributes() const noexcept { return attributes_; }
    void attributes(connector_attributes value) noexcept { attributes_ = value; }

private:
    node &owner_;
    std::string name_;
    datatype_t type_;
    shape_t shape_;
    connector_attributes attributes_ = cnctr_attr_none;
};

class NNCASE_API input_connector : public base_connector
{
public:
    using base_connector::base_connector;

    output_connector *connection() const noexcept { return connection_; }
    void connect(output_connector &connector);
    void clear_connection();

private:
    output_connector *connection_ = nullptr;
};

class NNCASE_API output_connector : public base_connector
{
public:
    template <class TName, class TShape>
    output_connector(node &owner, TName &&name, datatype_t type, TShape &&shape, memory_location_t memory_location = mem_data)
        : base_connector(owner, std::forward<TName>(name), type, std::forward<TShape>(shape)), memory_location_(memory_location)
    {
    }

    std::span<input_connector *const> connections() const noexcept { return connections_; }
    void connect(input_connector &connector);
    void disconnect(input_connector &connector);
    void clear_connections();
    // connector_attributes attributes() const noexcept { return attributes_; }
    // void attributes(connector_attributes value) noexcept { attributes_ = value; }
    memory_location_t memory_location() const noexcept { return memory_location_; }
    void memory_location(memory_location_t value) noexcept { memory_location_ = value; }

private:
    std::vector<input_connector *> connections_;
    // connector_attributes attributes_ = cnctr_attr_none;
    memory_location_t memory_location_;
};

/** @brief Operator node */
class NNCASE_API op_node : public expr_node
{
public:
    DEFINE_NODE_NODEKIND(node_op);

    op_node(opcode_t opcode);

    const type_t &type() override;

    /** @brief Get the return type of the function expression */
    const type_t &return_type() const noexcept { return return_type_; }

    /** @brief Get the parameters of the function expression */
    std::span<const var> parameters() const noexcept { return parameters_; }

    /** @brief Get the body of the function expression */
    const expr &body() const noexcept { return body_; }
    /** @brief Set the body of the function expression */
    void body(expr value) noexcept { body_ = std::move(value); }

private:
    type_t return_type_;
    std::vector<var> parameters_;
    expr body_;
};

using op = expr_t<op_node>;
}
