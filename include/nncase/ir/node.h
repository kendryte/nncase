/* Copyright 2020 Canaan Inc.
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
#include "connectors.h"
#include "opcode.h"
#include <list>
#include <span>
#include <unordered_map>

namespace nncase::ir
{
#define DEFINE_NODE_OPCODE(value)                                    \
    static constexpr node_opcode opcode() noexcept { return value; } \
    const node_opcode &runtime_opcode() const noexcept override { return value; }

class NNCASE_API node
{
public:
    node(std::string name = "")
        : name_(name) { }
    node(node &) = delete;
    node &operator=(node &) = delete;
    virtual ~node();

    const std::string &name() const noexcept { return name_; }

    template <class TArg, class... TArgs>
    void name(TArg arg, TArgs... args) { name_.assign(std::forward<TArg>(arg), std::forward<TArgs>(args)...); }

    std::span<input_connector *const> inputs() const noexcept { return input_connectors_; }
    std::span<output_connector *const> outputs() const noexcept { return output_connectors_; }

    input_connector &input_at(size_t index) const { return *input_connectors_.at(index); }
    output_connector &output_at(size_t index) const { return *output_connectors_.at(index); }

    virtual const node_opcode &runtime_opcode() const noexcept = 0;
    node_attributes attributes() const noexcept { return attributes_; }
    void attributes(node_attributes value) noexcept { attributes_ = value; }

    bool equals(node &other) const;

protected:
    template <class TName, class TShape>
    input_connector &add_input(TName &&name, datatype_t type, TShape &&shape)
    {
        auto ptr = input_connectors_storage_.emplace_back(std::make_unique<input_connector>(*this, std::forward<TName>(name), type, std::forward<TShape>(shape))).get();
        input_connectors_.emplace_back(ptr);
        return *ptr;
    }

    template <class TName, class TShape>
    output_connector &add_output(TName &&name, datatype_t type, TShape &&shape)
    {
        auto ptr = output_connectors_storage_.emplace_back(std::make_unique<output_connector>(*this, std::forward<TName>(name), type, std::forward<TShape>(shape))).get();
        output_connectors_.emplace_back(ptr);
        return *ptr;
    }

    virtual bool properties_equal(node &other) const = 0;

private:
    std::string name_;
    node_attributes attributes_ = node_attributes::node_attr_action;
    std::vector<input_connector *> input_connectors_;
    std::vector<output_connector *> output_connectors_;
    std::vector<std::unique_ptr<input_connector>> input_connectors_storage_;
    std::vector<std::unique_ptr<output_connector>> output_connectors_storage_;
};
}
