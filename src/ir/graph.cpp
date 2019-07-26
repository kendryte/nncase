#include <ir/graph.h>
#include <ir/visitor.h>
#include <unordered_set>

using namespace nncase;
using namespace nncase::ir;

namespace
{
class mark_visitor : public dfs_ir_visitor
{
public:
    using dfs_ir_visitor::visit;
    std::unordered_set<node *> used_nodes;

protected:
    bool visit(node &node) override
    {
        used_nodes.emplace(&node);
        return false;
    }
};
}

void graph::collect()
{
    mark_visitor visitor;
    visitor.visit(*this);

    auto end = std::remove_if(std::begin(nodes_), std::end(nodes_), [&](auto &node) {
        if (!visitor.used_nodes.contains(node.get()))
        {
            for (auto &in : node->inputs())
                in.clear_connection();
            for (auto &out : node->outputs())
                out.clear_connections();
            return true;
        }

        return false;
    });
    nodes_.erase(end, std::end(nodes_));
}
