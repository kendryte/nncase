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
#include <hlir/ops/constant.h>
#include <hlir/ops/dequantize.h>
#include <hlir/ops/fused_unary.h>
#include <hlir/ops/quantize.h>
#include <hlir/ops/table_lookup.h>
#include <hlir/transforms/neutral/fused_unary_to_lookup1d.h>
#include <hlir/visitor.h>
#include <kernels/neutral/neutral_kernels.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

namespace
{
std::vector<uint8_t> generate_lut(fused_unary &fu, const quant_param_t &iq, const quant_param_t &yq)
{
    const auto xf_min = (0 - iq.zero_point) / iq.scale;
    const auto xf_max = (255 - iq.zero_point) / iq.scale;

    const auto samples_count = 256;
    const auto sample_step = (xf_max - xf_min) / (samples_count - 1);
    std::array<float, samples_count> samples_x, samples_y;
    for (int32_t i = 0; i < samples_count; i++)
        samples_x[i] = xf_min + i * sample_step;

    std::stringstream ss;
    runtime::binary_writer bw(ss);
    runtime::nnil_builder builder(bw);

    fused_unary::compile_graph(fu.subgraph(), builder);
    auto buf = ss.str();
    std::vector<uint8_t> body(reinterpret_cast<uint8_t *>(buf.data()), reinterpret_cast<uint8_t *>(buf.data() + buf.size()));
    kernels::neutral::nnil_unary_method(samples_x.data(), samples_y.data(), samples_count, body);

    std::vector<uint8_t> lut(samples_count);
    for (int32_t i = 0; i < samples_count; i++)
        lut[i] = std::clamp((int32_t)std::round(samples_y[i] * yq.scale + yq.zero_point), 0, 255);
    return lut;
}
}

bool fused_unary_to_lookup1d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto fu = node_cast<fused_unary>(node))
    {
        if (fu->input().connection()->attributes() & cnctr_attr_need_quantize
            && fu->output().attributes() & cnctr_attr_need_quantize)
        {
            context.inputs.emplace_back(&fu->input());
            context.outputs.emplace_back(&fu->output());

            context.matched_nodes.emplace_back(&node);
            return true;
        }
    }

    return false;
}

void fused_unary_to_lookup1d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_fu = static_cast<fused_unary &>(*context.matched_nodes[0]);

    auto iq_p = quantizer_.get_quant_param(quantizer_.get(output), 8);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_fu.output()), 8);

    auto q = context.graph.emplace<quantize>(output.shape(), iq_p);
    q->name(output.owner().name() + "/quantize");
    auto table = context.graph.emplace<constant>(dt_uint8, shape_t { 256 }, generate_lut(old_fu, iq_p, yq_p));
    table->name(old_fu.name() + "/table");
    auto lut = context.graph.emplace<table_lookup1d>(dt_uint8, q->output().shape(), 256);
    lut->name(old_fu.name());
    auto deq = context.graph.emplace<dequantize>(old_fu.output().shape(), yq_p);
    deq->name(old_fu.name() + "/dequantize");
    link(old_fu.output(), deq->output(), &quantizer_);
    lut->input().connect(q->output());
    lut->table().connect(table->output());
    deq->input().connect(lut->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
