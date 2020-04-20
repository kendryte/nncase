/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/conv2d.h>
#include <hlir/ops/matmul.h>
#include <hlir/quantizer.h>
#include <hlir/transforms/neutral/add_quant_checkpoints.h>
#include <hlir/visitor.h>
#include <targets/target.h>
#include <xtensor/xstrides.hpp>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool add_quant_checkpoints_transform::on_try_match(node &node, transform_context &context)
{
    if (opcodes_.find(node.runtime_opcode()) != opcodes_.end())
    {
        if ((node.runtime_opcode() == op_binary || node.runtime_opcode() == op_matmul
                || node.runtime_opcode() == op_conv2d)
            && xt::compute_size(node.output_at(0).shape()) < context.target.options().output_quantize_threshold)
        {
            return false;
        }
        else if (auto conv = node_cast<conv2d>(node))
        {
            auto weights = conv->weights();
            auto total_range = quantizer::fixup_range(quantizer::get_range(weights.begin(), weights.end()));
            if (total_range.max - total_range.min > context.target.options().weights_quantize_threshold)
                return false;
        }
        else if (auto mm = node_cast<matmul>(node))
        {
            if (auto w = try_get_direct_parent<constant>(node))
            {
                auto w_beg = reinterpret_cast<const float *>(w->data().data());
                auto w_end = w_beg + w->data().size() / sizeof(float);

                auto total_range = quantizer::fixup_range(quantizer::get_range(w_beg, w_end));
                if (total_range.max - total_range.min > context.target.options().weights_quantize_threshold)
                    return false;
            }
        }

        bool not_processed = false;
        if (!node.inputs().empty()
            && std::any_of(node.inputs().begin(), node.inputs().end(), [](input_connector &in) { return (in.connection()->attributes() & cnctr_attr_need_quantize) != cnctr_attr_need_quantize; }))
            not_processed = true;
        if (!not_processed
            && !node.outputs().empty()
            && std::any_of(node.outputs().begin(), node.outputs().end(), [](output_connector &out) { return (out.attributes() & cnctr_attr_need_quantize) != cnctr_attr_need_quantize; }))
            not_processed = true;

        if (not_processed)
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
    node.attributes(node.attributes() | node_attr_need_quantize);

    for (size_t i = 0; i < node.inputs().size(); i++)
    {
        auto &output = *node.input_at(i).connection();
        output.attributes(output.attributes() | cnctr_attr_need_quantize);
    }

    for (size_t i = 0; i < node.outputs().size(); i++)
    {
        auto &output = node.output_at(i);
        output.attributes(output.attributes() | cnctr_attr_need_quantize);
    }
}
