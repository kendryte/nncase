#pragma once
#include "graph.h"
#include <unordered_set>

namespace nncase
{
namespace ir
{
    class ir_visitor
    {
    public:
        void visit(graph &graph);
        void visit(xtl::span<ir::output_node *> outputs);

        bool visited(node &node) const noexcept;

    protected:
        void mark_visit(node &node);

        virtual bool visit_strategry(node &node) = 0;

        virtual bool visit(node &node);

    private:
        std::unordered_set<node *> visited_;
    };

    class dfs_ir_visitor : public ir_visitor
    {
    protected:
        virtual bool visit_strategry(node &node) final override;

    private:
    };
}
}
