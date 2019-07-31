#include <ir/ops/dequantize.h>
#include <ir/ops/quantize.h>
#include <transforms/neutral/fold_quantize.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool fold_quantize_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_quantize)
    {
        auto &q = static_cast<quantize &>(node);
        for (auto &&conn : q.output().connections())
        {
            if (conn->owner().runtime_opcode() == op_dequantize)
            {
                auto &deq = static_cast<dequantize &>(conn->owner());

                if (almost_equal(q.quant_param(), deq.quant_param()))
                {
                    context.inputs.emplace_back(&q.input());
                    context.outputs.emplace_back(&deq.output());

                    context.matched_nodes.emplace_back(&q);
                    context.matched_nodes.emplace_back(&deq);
                    return true;
                }
            }
        }
    }
    else if (node.runtime_opcode() == op_dequantize)
    {
        auto &deq = static_cast<dequantize &>(node);
        for (auto &&conn : deq.output().connections())
        {
            if (conn->owner().runtime_opcode() == op_quantize)
            {
                auto &q = static_cast<quantize &>(conn->owner());

                if (almost_equal(q.quant_param(), deq.quant_param()))
                {
                    context.inputs.emplace_back(&deq.input());
                    context.outputs.emplace_back(&q.output());

                    context.matched_nodes.emplace_back(&deq);
                    context.matched_nodes.emplace_back(&q);
                    return true;
                }
            }
        }
    }

    return false;
}

void fold_quantize_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
