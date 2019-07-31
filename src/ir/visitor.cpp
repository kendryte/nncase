#include <ir/visitor.h>

using namespace nncase;
using namespace nncase::ir;

void ir_visitor::visit(graph &graph)
{
    visit(graph.outputs());
}

void ir_visitor::visit(xtl::span<ir::output_node *> outputs)
{
    visited_.clear();

    for (auto &&out : outputs)
    {
        if (visit_strategry(*out))
            return;
    }
}

bool ir_visitor::visited(node &node) const noexcept
{
    return visited_.contains(&node);
}

void ir_visitor::mark_visit(node &node)
{
    visited_.emplace(&node);
}

bool ir_visitor::visit(node &node)
{
    return false;
}

bool dfs_ir_visitor::visit_strategry(node &node)
{
    if (!visited(node))
    {
        mark_visit(node);

        for (auto &&in : node.inputs())
        {
            if (in.connection())
            {
                if (visit_strategry(in.connection()->owner()))
                    return true;
            }
        }

        if (visit(node))
            return true;

        for (auto &&out : node.outputs())
        {
            for (auto &&in : out.connections())
            {
                auto &owner = in->owner();
                if (owner.attributes() & node_attr_action)
                {
                    if (visit_strategry(owner))
                        return true;
                }
            }
        }
    }

    return false;
}
