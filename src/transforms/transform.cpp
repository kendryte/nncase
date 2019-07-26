#include <algorithm>
#include <ir/visitor.h>
#include <transforms/transform.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool transform::try_match(node &node, transform_context &context)
{
    if (on_try_match(node, context))
    {
        if (!skip_self_contained_check())
        {
            for (auto &&node : context.matched_nodes)
            {
                // there exist input connectors out of the subgraph
                for (auto &&in : node->inputs())
                {
                    if (in.connection())
                    {
                        if (std::find(std::begin(context.inputs), std::end(context.inputs), &in) == std::end(context.inputs)
                            && std::find(std::begin(context.matched_nodes), std::end(context.matched_nodes), &in.connection()->owner()) == std::end(context.matched_nodes))
                            return false;
                    }
                }

                // there exist output connectors out of the subgraph
                for (auto &&out : node->outputs())
                {
                    for (auto &&conn : out.connections())
                    {
                        if (std::find(std::begin(context.outputs), std::end(context.outputs), &out) == std::end(context.outputs)
                            && std::find(std::begin(context.matched_nodes), std::end(context.matched_nodes), &conn->owner()) == std::end(context.matched_nodes))
                            return false;
                    }
                }
            }
        }

        return true;
    }

    return false;
}

bool transform::skip_self_contained_check() const noexcept
{
    return false;
}

namespace
{
class transform_apply_visitor : public dfs_ir_visitor
{
public:
    using dfs_ir_visitor::visit;
    graph *graph;
    bool need_retry = false;
    transform *transform;

protected:
    bool visit(node &node) override
    {
        transform_context context { *graph };
        if (transform->try_match(node, context))
        {
            transform->process(context);
            need_retry = true;
            return true;
        }

        return false;
    }
};
}

void nncase::transforms::transform_graph(graph &graph, xtl::span<transform *> transforms)
{
    transform_apply_visitor visitor;
    visitor.graph = &graph;
    bool next_pass = false;

    do
    {
        next_pass = false;

        for (auto &&transform : transforms)
        {
            visitor.transform = transform;
            visitor.need_retry = false;
            visitor.visit(graph);

            if (visitor.need_retry)
            {
                next_pass = true;
                graph.collect();
                break;
            }
        }
    } while (next_pass);
}

std::vector<ir::input_connector *> nncase::transforms::dup(xtl::span<ir::input_connector *const> connections)
{
    std::vector<ir::input_connector *> con;
    std::copy(connections.begin(), connections.end(), std::back_inserter(con));
    return con;
}
