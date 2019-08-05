#include <ir/ops/k210/kpu_data_exchange.h>
#include <ir/visitor.h>
#include <transforms/k210/fold_kpu_upload.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

bool fold_kpu_upload_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_kpu_upload)
    {
        auto &up = static_cast<kpu_upload &>(node);
        if (auto down = try_get_direct_child<kpu_download>(up))
        {
            context.inputs.emplace_back(&up.input());
            context.outputs.emplace_back(&down->output());

            context.matched_nodes.emplace_back(&up);
            context.matched_nodes.emplace_back(down);
            return true;
        }
    }
    else if (node.runtime_opcode() == op_k210_kpu_download)
    {
        auto &down = static_cast<kpu_download &>(node);
        if (auto up = try_get_direct_child<kpu_upload>(down))
        {
            context.inputs.emplace_back(&down.input());
            context.outputs.emplace_back(&up->output());

            context.matched_nodes.emplace_back(&down);
            context.matched_nodes.emplace_back(up);
            return true;
        }
    }

    return false;
}

void fold_kpu_upload_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}

bool fold_input_kpu_upload_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_input_node)
    {
        auto &in = static_cast<input_node &>(node);
        if (auto up = try_get_direct_child<kpu_upload>(in))
        {
            context.outputs.emplace_back(&up->output());

            context.matched_nodes.emplace_back(&in);
            context.matched_nodes.emplace_back(up);
            return true;
        }
    }

    return false;
}

void fold_input_kpu_upload_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto &old_in = static_cast<input_node &>(*context.matched_nodes[0]);

    auto input = context.graph.emplace<input_node>(dt_uint8, old_in.output().shape(), mem_k210_kpu);

    for (auto &in : dup(inputs))
        in->connect(input->output());
}
