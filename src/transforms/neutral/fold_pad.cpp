#include <ir/ops/pad.h>
#include <ir/visitor.h>
#include <transforms/neutral/fold_pad.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool fold_nop_pad_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_pad)
    {
        auto &p = static_cast<pad &>(node);

        if (std::all_of(p.paddings().begin(), p.paddings().end(), [](const padding &value) { return value == padding::zero(); }))
        {
            context.inputs.emplace_back(&p.input());
            context.outputs.emplace_back(&p.output());

            context.matched_nodes.emplace_back(&p);
            return true;
        }
    }

    return false;
}

void fold_nop_pad_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
