/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/codegen/binary_writer.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/table_lookup.h>
#include <nncase/ir/visitor.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/nnil.h>
#include <nncase/transforms/neutral/fused_unary_to_lookup1d.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

namespace
{
std::vector<uint8_t> generate_lut(fused_unary &fu, const quant_param_t &iq, const quant_param_t &yq)
{
    const auto range = iq.range<uint8_t>();
    const auto samples_count = 256;
    const auto sample_step = range.length() / (samples_count - 1);
    std::array<float, samples_count> samples_x, samples_y;
    for (int32_t i = 0; i < samples_count; i++)
        samples_x[i] = range.min + i * sample_step;

    std::stringstream ss;
    binary_writer bw(ss);
    codegen::nnil_builder builder(bw);

    fused_unary::compile_graph(fu.subgraph(), builder);
    auto buf = ss.str();
    std::vector<gsl::byte> body(reinterpret_cast<gsl::byte *>(buf.data()), reinterpret_cast<gsl::byte *>(buf.data() + buf.size()));
    kernels::nnil_unary_method(samples_x.data(), samples_y.data(), samples_count, body)
        .unwrap_or_throw();

    std::vector<uint8_t> lut(samples_count);
    for (int32_t i = 0; i < samples_count; i++)
        lut[i] = kernels::detail::quantize<uint8_t>(samples_y[i], yq);
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

    auto &quantizer = *context.quantizer;
    auto iq_p = quantizer.get_quant_param(quantizer.get(output), 8, quantizer::quant_mode::unsigned_mode);
    auto yq_p = quantizer.get_quant_param(quantizer.get(old_fu.output()), 8, quantizer::quant_mode::unsigned_mode);

    auto q = context.graph.emplace<quantize>(output.type(), output.shape(), dt_uint8, iq_p);
    q->name(output.owner().name() + "/quantize");
    auto table = context.graph.emplace<constant>(dt_uint8, shape_t { 256 }, generate_lut(old_fu, iq_p, yq_p));
    table->name(old_fu.name() + "/table");
    auto lut = context.graph.emplace<table_lookup1d>(dt_uint8, q->output().shape(), 256);
    lut->name(old_fu.name());
    auto deq = context.graph.emplace<dequantize>(dt_uint8, old_fu.output().shape(), output.type(), yq_p);
    deq->record_output_connectors_quant_map(deq->output_at(0), old_fu.output_at(0));
    deq->record_node_name_before_quant(old_fu.name());
    deq->name(old_fu.name() + "/dequantize");
    link(old_fu.output(), deq->output(), &quantizer);
    lut->input().connect(q->output());
    lut->table().connect(table->output());
    deq->input().connect(lut->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
