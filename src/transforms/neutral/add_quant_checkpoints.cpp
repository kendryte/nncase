#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/visitor.h>
#include <transforms/neutral/add_quant_checkpoints.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool add_quant_checkpoints_transform::on_try_match(node &node, transform_context &context)
{
    if (opcodes_.find(node.runtime_opcode()) != opcodes_.end())
    {
        if (!try_get_direct_child<fake_dequantize>(node))
        {
            for (auto &in : node.inputs())
                context.inputs.emplace_back(&in);
            for (auto &out : node.outputs())
                context.outputs.emplace_back(&out);

            context.matched_nodes.emplace_back(&node);
            return true;
        }
    }

    return false;
}

void add_quant_checkpoints_transform::process(transform_context &context)
{
    auto &node = *context.matched_nodes[0];

    for (size_t i = 0; i < node.inputs().size(); i++)
    {
        auto &output = *node.input_at(i).connection();
        auto q = context.graph.emplace<fake_quantize>(output.shape());
        q->input().connect(output);
        node.input_at(i).connect(q->output());
    }

    for (size_t i = 0; i < node.outputs().size(); i++)
    {
        auto &output = node.output_at(i);
        auto inputs = dup(output.connections());
        auto deq = context.graph.emplace<fake_dequantize>(output.shape());
        deq->input().connect(output);

        for (auto &in : inputs)
            in->connect(deq->output());
    }
}
