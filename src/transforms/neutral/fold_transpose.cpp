#include <ir/ops/transpose.h>
#include <transforms/neutral/fold_transpose.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

// Transpose (perm = p1)
//     |
//     v
// Transpose (perm = p2)
//
// p1[p2[i]] == i

bool fold_transpose_transform::on_try_match(node &node, transform_context &context)
{
    if (node.opcode() == op_transpose)
    {
        auto &tp1 = static_cast<transpose &>(node);
        for (auto &&conn : tp1.output().connections())
        {
            if (conn->owner().opcode() == op_transpose)
            {
                auto &tp2 = static_cast<transpose &>(conn->owner());

                if (tp1.perm().size() == tp2.perm().size())
                {
                    for (size_t i = 0; i < tp1.perm().size(); i++)
                    {
                        if (tp1.perm()[tp2.perm()[i]] != i)
                            return false;
                    }

                    context.inputs.emplace_back(&tp1.input());
                    context.outputs.emplace_back(&tp2.output());

                    context.matched_nodes.emplace_back(&tp1);
                    context.matched_nodes.emplace_back(&tp2);
                    return true;
                }
            }
        }
    }

    return false;
}

void fold_transpose_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    for (auto &in : dup(inputs))
        in->connect(output);
}
