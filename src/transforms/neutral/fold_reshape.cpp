#include <ir/ops/reshape.h>
#include <ir/visitor.h>
#include <transforms/neutral/fold_reshape.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool fold_reshape_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_reshape)
    {
        auto &rp1 = static_cast<reshape &>(node);
        if (auto rp2 = try_get_direct_child<reshape>(rp1))
        {
            context.inputs.emplace_back(&rp1.input());
            context.outputs.emplace_back(&rp2->output());

            context.matched_nodes.emplace_back(&rp1);
            context.matched_nodes.emplace_back(rp2);
            return true;
        }
    }

    return false;
}

void fold_reshape_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &rp2 = *static_cast<reshape *>(context.matched_nodes[1]);

    auto rp = context.graph.emplace<reshape>(output.type(), output.shape(), rp2.output().shape());

    rp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(rp->output());
}

bool fold_nop_reshape_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_reshape)
    {
        auto &rp = static_cast<reshape &>(node);

        if (rp.input().shape() == rp.output().shape())
        {
            context.inputs.emplace_back(&rp.input());
            context.outputs.emplace_back(&rp.output());

            context.matched_nodes.emplace_back(&rp);
            return true;
        }
    }

    return false;
}

void fold_nop_reshape_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
