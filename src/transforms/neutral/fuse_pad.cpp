/* Copyright 2019 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ir/ops/conv2d.h>
#include <ir/ops/pad.h>
#include <ir/visitor.h>
#include <transforms/neutral/fuse_pad.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::transforms;

bool fuse_pad_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_pad)
    {
        auto &p = static_cast<pad &>(node);
        if (p.paddings().size() == 4 && p.paddings()[2].before >= 0 && p.paddings()[2].after >= 0
            && p.paddings()[3].before >= 0 && p.paddings()[3].after >= 0
            && (p.paddings()[2].sum() > 0 || p.paddings()[3].sum() > 0))
        {
            if (auto conv = try_get_direct_child<conv2d>(p))
            {
                context.inputs.emplace_back(&p.input());
                context.outputs.emplace_back(&conv->output());

                context.matched_nodes.emplace_back(&p);
                context.matched_nodes.emplace_back(conv);
                return true;
            }
        }
    }

    return false;
}

void fuse_pad_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_p = static_cast<pad &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[1]);

    auto paddings = old_p.paddings();
    std::array<padding, 2> conv_paddings {
        old_conv.padding_h(),
        old_conv.padding_w()
    };

    for (size_t i = 2; i < 4; i++)
    {
        auto &src = paddings[i];
        auto &dest = conv_paddings[i - 2];
        if (src.before > 0)
        {
            dest.before += src.before;
            src.before = 0;
        }
        if (src.after > 0)
        {
            dest.after += src.after;
            src.after = 0;
        }
    }

    auto p = context.graph.emplace<pad>(old_p.output().type(), output.shape(), paddings, old_p.pad_value());
    auto conv = context.graph.emplace<conv2d>(p->output().shape(), old_conv.weights(), old_conv.bias(), old_conv.groups(),
        conv_paddings[0], conv_paddings[1], old_conv.stride_h(), old_conv.stride_w(), old_conv.dilation_h(), old_conv.dilation_w(), old_conv.fused_activation());
    conv->input().connect(p->output());

    p->input().connect(output);

    for (auto &in : dup(inputs))
        in->connect(conv->output());
}
