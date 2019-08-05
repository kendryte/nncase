/* Copyright 2019 Canaan Inc.
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
#include "ir_types.h"
#include <optional>
#include <string>
#include <vector>
#include <xtensor/xshape.hpp>
#include <xtl/xspan.hpp>

namespace nncase
{
namespace ir
{
    class node;
    class output_connector;

    class base_connector
    {
    public:
        template <class TName, class TShape>
        base_connector(node &owner, TName &&name, datatype_t type, TShape &&shape)
            : owner_(owner), name_(std::forward<TName>(name)), type_(type), shape_(std::forward<TShape>(shape))
        {
        }

        node &owner() const noexcept { return owner_; }
        const std::string &name() const noexcept { return name_; }
        datatype_t type() const noexcept { return type_; }
        const shape_t &shape() const noexcept { return shape_; }

    private:
        node &owner_;
        std::string name_;
        datatype_t type_;
        shape_t shape_;
    };

    class input_connector : public base_connector
    {
    public:
        using base_connector::base_connector;

        output_connector *connection() const noexcept { return connection_; }
        void connect(output_connector &connector);
        void clear_connection();

    private:
        output_connector *connection_ = nullptr;
    };

    class output_connector : public base_connector
    {
    public:
        template <class TName, class TShape>
        output_connector(node &owner, TName &&name, datatype_t type, TShape &&shape, memory_type_t memory_type)
            : base_connector(owner, std::forward<TName>(name), type, std::forward<TShape>(shape)), memory_type_(memory_type)
        {
        }

        xtl::span<input_connector *const> connections() const noexcept { return connections_; }
        void connect(input_connector &connector);
        void disconnect(input_connector &connector);
        void clear_connections();
        memory_type_t memory_type() const noexcept { return memory_type_; }

    private:
        std::vector<input_connector *> connections_;
        memory_type_t memory_type_;
    };
}
}
