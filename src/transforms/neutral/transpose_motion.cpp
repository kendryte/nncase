#include <ir/ops/concat.h>
#include <ir/ops/transpose.h>
#include <transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

// Transpose (perm = p1)
//     |
//     v
// concat (perm = p2)
//
// p1[p2[i]] == i

bool transpose_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (node.opcode() == op_concat)
    {
        auto &con = static_cast<concat &>(node);

        context.matched_nodes.emplace_back(&con);
        context.outputs.emplace_back(&con.output());

        axis_t perm;
        for (auto &&conn : con.inputs())
        {
            if (conn.connection()->owner().opcode() == op_transpose)
            {
                auto &tp = static_cast<transpose &>(conn.connection()->owner());

                if (perm.empty())
                    perm = tp.perm();

                if (perm != tp.perm())
                    return false;

                context.inputs.emplace_back(&tp.input());
                context.matched_nodes.emplace_back(&tp);
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    return false;
}

void transpose_motion_transform::process(transform_context &context)
{
    auto &con = static_cast<concat &>(*context.matched_nodes[0]);
    auto &perm = static_cast<transpose &>(*context.matched_nodes[1]).perm();

    std::vector<shape_t> new_in_shapes;
    for (auto &&in : context.inputs)
        new_in_shapes.emplace_back(in->shape());

    auto new_axis = perm[con.axis()];
    auto new_con = context.graph.emplace<concat>(con.output().type(), new_in_shapes, new_axis);
    auto new_trans = context.graph.emplace<transpose>(con.output().type(), new_con->output().shape(), perm);
    new_trans->input().connect(new_con->output());

    for (size_t i = 0; i < context.inputs.size(); i++)
        new_con->input_at(i).connect(*context.inputs[i]->connection());
    for (auto &&out : dup(con.output().connections()))
        out->connect(new_trans->output());
}
