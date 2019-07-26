#pragma once
#include <ir/graph.h>
#include <vector>

namespace nncase
{
namespace transforms
{
    struct transform_context
    {
        ir::graph &graph;
        std::vector<ir::node *> matched_nodes;
        std::vector<ir::input_connector *> inputs;
        std::vector<ir::output_connector *> outputs;
    };

    class transform
    {
    public:
        bool try_match(ir::node &node, transform_context &context);

        virtual void process(transform_context &context) = 0;

    protected:
        virtual bool skip_self_contained_check() const noexcept;
        virtual bool on_try_match(ir::node &node, transform_context &context) = 0;
    };

    void transform_graph(ir::graph &graph, xtl::span<transform *> transforms);
    std::vector<ir::input_connector *> dup(xtl::span<ir::input_connector *const> connections);
}
}
