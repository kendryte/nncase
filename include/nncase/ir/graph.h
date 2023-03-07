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
#include "node.h"
#include "placeholders.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace nncase::ir
{
class graph;

struct split_graph_result
{
    std::unique_ptr<graph> subgraph;
    std::unordered_map<input_node *, output_connector *> inputs;
    std::unordered_map<output_node *, std::vector<input_connector *>> outputs;
};

class NNCASE_API graph
{
public:
    graph() noexcept;
    explicit graph(const module_type_t &module_type) noexcept
        : module_type_(module_type) { }

    graph(graph &) = delete;
    graph(graph &&) = delete;

    const std::string &name() const noexcept { return name_; }
    std::string escaped_name() const noexcept;
    void name(std::string value) { name_ = std::move(value); }
    const module_type_t &module_type() const noexcept { return module_type_; }
    void set_module_type(module_type_t type) { this->module_type_ = type; }

    std::span<std::unique_ptr<node>> nodes() noexcept { return nodes_; }
    std::span<input_node *> inputs() noexcept { return inputs_; }
    std::span<output_node *> outputs() noexcept { return outputs_; }
    std::span<std::unique_ptr<graph>> subgraphs() noexcept { return subgraphs_; }
    std::vector<graph *> reachable_graphs() noexcept;

    std::span<std::unique_ptr<node> const> nodes() const noexcept { return nodes_; }
    std::span<input_node *const> inputs() const noexcept { return inputs_; }
    std::span<output_node *const> outputs() const noexcept { return outputs_; }
    std::span<std::unique_ptr<graph> const> subgraphs() const noexcept { return subgraphs_; }

    template <class T, class... TArgs>
    T *emplace(TArgs &&...args)
    {
        auto node = static_cast<T *>(nodes_.emplace_back(new T(std::forward<TArgs>(args)...)).get());
        if constexpr (std::is_same_v<T, input_node>)
            inputs_.emplace_back(node);
        else if constexpr (std::is_same_v<T, output_node>)
            outputs_.emplace_back(node);
        return node;
    }

    void assign_names();
    void dce();
    void cse();
    void merge_module_regions();
    split_graph_result split_subgraph(std::span<node *const> nodes, bool reorder_input = false);
    graph &add_subgraph(std::unique_ptr<graph> subgraph);

private:
    std::string name_;
    module_type_t module_type_;
    std::vector<std::unique_ptr<node>> nodes_;
    std::vector<std::unique_ptr<graph>> subgraphs_;
    std::vector<input_node *> inputs_;
    std::vector<output_node *> outputs_;
};
}
