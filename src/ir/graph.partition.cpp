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
#include <queue>
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
    std::unordered_map<output_connector *, int> need_remove_outputs;

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
                {
                    if (need_remove_outputs.find((*it)->connection()) != need_remove_outputs.end())
                        need_remove_outputs.at((*it)->connection()) -= 1;
                    else
                        need_remove_outputs.emplace((*it)->connection(),
                            (*it)->connection()->connections().size() - 1);
                    it = region_inputs.erase(it);
                }
                else
                    ++it;
            }

            for (auto it = need_remove_outputs.begin(); it != need_remove_outputs.end();)
            {
                if (it->second == 0)
                {
                    outputs.erase(it->first);
                    it = need_remove_outputs.erase(it);
                }
                else
                    it++;
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

typedef struct Region_node
{
    std::list<region>::iterator node;
    Region_node *parent = nullptr;
    Region_node *child = nullptr;
    Region_node *bro = nullptr;
} Region_node, *Region_Tree;

class Region_tree
{
public:
    Region_tree(std::list<region> &rg)
        : regions_(rg) { }
    Region_node *create_tree(std::list<region>::iterator new_node, int depth)
    {

        Region_node *root = create_node();
        root->node = new_node;

        // find a path from itb--> ita
        if (new_node == target_region_)
        {
            leaves_.push_back(root);
            return root;
        }

        // limit tree depth
        if (depth >= 10)
        {
            skip_ = true;
            return root;
        }

        for (auto it : new_node->region_inputs)
        {
            for (auto itb = regions_.begin(); itb != regions_.end(); itb++)
            {
                if (itb->outputs.contains(it->connection()))
                {
                    if (root->child == nullptr)
                    {
                        root->child = create_tree(itb, depth + 1);
                        root->child->parent = root;
                    }
                    else
                    {
                        root->bro = create_tree(itb, depth);
                        root->bro->parent = root;
                        root->bro = root->bro->bro;
                    }
                }
            }
        }

        return root;
    }

    bool not_have_circle()
    {
        // if tree depth > 10, ignore merge itb--> ita
        if (skip_)
            return false;
        // each leaf has only one path to root.
        // if all the paths of leaves to root don't have CPU op ,itb can merge to ita.
        for (auto it : leaves_)
        {
            auto condition_ptr = it->parent;
            if (condition_ptr->node == start_region_)
                continue;
            while (condition_ptr != nullptr)
            {
                if (condition_ptr->node->module_type == runtime::stackvm::stackvm_module_type && !condition_ptr->node->is_all_noaction)
                    return false;
                condition_ptr = condition_ptr->parent;
            }
        }
        return true;
    }

    void set_label_region(std::list<region>::iterator ita, std::list<region>::iterator itb)
    {
        start_region_ = itb;
        target_region_ = ita;
    }

    void free_tree(Region_node *root)
    {
        if (root != nullptr)
        {
            free_tree(root->child);
            free_tree(root->bro);
            delete root;
        }
    }

private:
    Region_node *create_node()
    {
        Region_Tree node = new Region_node();
        return node;
    }

    std::list<region>::iterator start_region_;
    std::list<region>::iterator target_region_;
    std::vector<Region_node *> leaves_;
    bool skip_;
    std::list<region> &regions_;
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
        auto creator = make_relay_ir_visitor([this](node &node) {
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

                if (conn.runtime_opcode() == op_constant)
                {
                    last_region = nullptr;
                    break;
                }

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
            changed |= merge_parent_region();
            changed |= merge_same_input_region();

        } while (changed);

        do
        {
            changed = false;
            changed |= merge_child_region_stage_2();
        } while (changed);
    }

    bool check_circle(std::list<region>::iterator ita, std::list<region>::iterator itb)
    {
        // merge directly
        bool merge_directly = true;
        for (auto it : ita->outputs)
        {
            if (std::all_of(it->connections().begin(), it->connections().end(),
                    [&](input_connector *out) {
                        return itb->region_inputs.contains(out);
                    }))
                continue;
            else
                merge_directly = false;
        }
        if (merge_directly)
            return true;

        if (itb->region_inputs.size() == 1)
        {
            return true;
        }

        auto check = std::make_shared<Region_tree>(regions_);
        check->set_label_region(ita, itb);
        auto root = check->create_tree(itb, 0);
        auto flag = check->not_have_circle();
        check->free_tree(root);
        return flag;
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
                        && std::all_of(itb->region_inputs.begin(), itb->region_inputs.end(), [&](input_connector *in) { return ita->outputs.contains(in->connection()); }))
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

    bool merge_child_region_stage_2()
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

                    // itb's has inputs connect to ita's output without circle
                    if ((ita->module_type == itb->module_type || itb->is_all_noaction)
                        && std::any_of(itb->region_inputs.begin(), itb->region_inputs.end(), [&](input_connector *in) { return ita->outputs.contains(in->connection()); })
                        && check_circle(ita, itb))
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

    bool merge_parent_region()
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

                    // itb's outputs all connect to ita's input
                    if (itb->is_all_noaction
                        && std::all_of(itb->outputs.begin(), itb->outputs.end(), [&](output_connector *out) { return std::all_of(out->connections().begin(), out->connections().end(), [&](input_connector *in) { return ita->region_inputs.contains(in); }); }))
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

    bool merge_same_input_region()
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

                    // itb has the same input with ita
                    if (ita->module_type == itb->module_type)
                    {
                        std::unordered_set<output_connector *> outputs_a, outputs_b;
                        for (auto in : ita->region_inputs)
                            outputs_a.emplace(in->connection());
                        for (auto in : itb->region_inputs)
                            outputs_b.emplace(in->connection());
                        if (std::all_of(outputs_a.begin(), outputs_a.end(), [&](output_connector *out) { return outputs_b.contains(out); })
                            && std::all_of(outputs_b.begin(), outputs_b.end(), [&](output_connector *out) { return outputs_a.contains(out); }))
                            to_be_merge.emplace_back(itb);
                    }
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
        if (node.module_type() != runtime::stackvm::stackvm_module_type)
            region.module_type = node.module_type();
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

        auto split = split_subgraph(region.nodes, true);
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
