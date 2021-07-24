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
#include <nncase/ir/graph.h>
#include <nncase/ir/ops/call.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;

namespace
{
struct region
{
    bool is_all_noaction = true;
    module_type_t module_type;
    std::vector<node *> nodes;
    std::unordered_set<node *> nodes_set;
    std::unordered_set<input_connector *> region_inputs;
    std::unordered_set<output_connector *> outputs;

    region(module_type_t module_type)
        : module_type(module_type)
    {
    }

    bool add_node(node &n)
    {
        if (nodes_set.emplace(&n).second)
        {
            nodes.emplace_back(&n);
            for (auto in : n.inputs())
                region_inputs.emplace(in);
            for (auto out : n.outputs())
                outputs.emplace(out);
            for (auto it = region_inputs.begin(); it != region_inputs.end();)
            {
                if (outputs.contains((*it)->connection()))
                    it = region_inputs.erase(it);
                else
                    ++it;
            }

            if (is_all_noaction && n.attributes() & node_attr_action)
                is_all_noaction = false;
            return true;
        }

        return false;
    }

    void embed_constant(graph &g)
    {
        std::vector<node *> to_add_constants;
        for (auto node : nodes)
        {
            for (auto in : node->inputs())
            {
                if (auto c = node_cast<constant>(in->connection()->owner()))
                {
                    // 1. If exclusive, add it to region
                    if (c->output().connections().size() == 1)
                    {
                        to_add_constants.emplace_back(c);
                    }
                    // 2. Create a new one
                    else
                    {
                        auto new_c = g.emplace<constant>(c->data_type(), c->output().shape(), c->data());
                        new_c->name(c->name());
                        new_c->alignment(c->alignment());
                        new_c->attributes(c->attributes());
                        to_add_constants.emplace_back(new_c);
                    }
                }
            }
        }

        for (auto c : to_add_constants)
            add_node(*c);
    }

    void merge(region &other)
    {
        for (auto node : other.nodes)
            add_node(*node);
    }
};

class graph_merger
{
public:
    graph_merger(graph &g)
        : g_(g)
    {
    }

    void merge()
    {
        create_regions();
        embed_constant();
        merge_regions();
    }

    const std::list<region> &regions() const noexcept { return regions_; }

private:
    void create_regions()
    {
        auto creator = make_relay_ir_visitor([this](node &node)
            {
                // 1. Skip in/out/const
                if (node.runtime_opcode() == op_input_node
                    || node.runtime_opcode() == op_output_node
                    || node.runtime_opcode() == op_constant)
                    return;

                // 2. Find last region
                region *last_region = nullptr;
                for (auto in : node.inputs())
                {
                    auto &conn = in->connection()->owner();
                    auto it = node_to_region_.find(&conn);
                    if (it != node_to_region_.end())
                    {
                        // 2.1. Last region not set, set it
                        if (!last_region)
                        {
                            last_region = it->second;
                        }
                        // 2.2. Last region set but different, create new region
                        else if (last_region != it->second)
                        {
                            last_region = nullptr;
                            break;
                        }
                    }
                }

                // 3. Last region not set or different module type, create new region
                if (!last_region || last_region->module_type != node.module_type())
                {
                    auto &region = regions_.emplace_back(node.module_type());
                    add_node_to_region(region, node);
                }
                // 4. Add to last region
                else
                {
                    add_node_to_region(*last_region, node);
                }
            });
        creator.visit(g_);
    }

    void embed_constant()
    {
        for (auto &region : regions_)
            region.embed_constant(g_);
        g_.dce();
    }

    void merge_regions()
    {
        bool changed;
        do
        {
            changed = false;
            changed |= merge_child_region();
        } while (changed);
    }

    bool merge_child_region()
    {
        bool ever_changed = false;
        bool changed;
        do
        {
            changed = false;
            for (auto ita = regions_.begin(); ita != regions_.end(); ++ita)
            {
                std::vector<std::list<region>::iterator> to_be_merge;
                for (auto itb = regions_.begin(); itb != regions_.end(); ++itb)
                {
                    // don't merge stackvm region
                    if (ita == itb
                        || (ita->module_type == runtime::stackvm::stackvm_module_type
                            && itb->module_type == runtime::stackvm::stackvm_module_type))
                        continue;

                    // itb's inputs all connect to ita's output
                    if ((ita->module_type == itb->module_type || itb->is_all_noaction)
                        && std::all_of(itb->region_inputs.begin(), itb->region_inputs.end(), [&](input_connector *in)
                            { return ita->outputs.contains(in->connection()); }))
                        to_be_merge.emplace_back(itb);
                }

                if (!to_be_merge.empty())
                {
                    for (auto region : to_be_merge)
                    {
                        ita->merge(*region);
                        regions_.erase(region);
                    }

                    changed = ever_changed = true;
                    break;
                }
            }
        } while (changed);
        return ever_changed;
    }

    void add_node_to_region(region &region, node &node)
    {
        region.add_node(node);
        node_to_region_.emplace(&node, &region);
    }

private:
    graph &g_;
    std::list<region> regions_;
    std::unordered_map<node *, region *> node_to_region_;
};
}

void graph::merge_module_regions()
{
    graph_merger merger(*this);
    merger.merge();

    std::unordered_map<std::string, size_t> subids;
    for (auto &region : merger.regions())
    {
        // Don't create subgraph for stackvm
        if (region.module_type == runtime::stackvm::stackvm_module_type)
            continue;

        auto split = split_subgraph(region.nodes);
        auto &subg = add_subgraph(std::move(split.subgraph));
        auto c = emplace<call>(subg);
        c->name(std::string(region.module_type.data()) + "_" + std::to_string(subids[region.module_type.data()]++));
        subg.name(c->name());

        for (auto &inp : split.inputs)
        {
            auto &outer_in = c->outer_connector(*inp.first);
            outer_in.connect(*inp.second);
        }

        for (auto &outp : split.outputs)
        {
            auto &outer_out = c->outer_connector(*outp.first);
            for (auto in : outp.second)
                in->connect(outer_out);
        }
    }
}
