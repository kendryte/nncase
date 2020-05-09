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
#include <algorithm>
#include <hlir/ops/dequantize.h>
#include <hlir/ops/fused_unary.h>
#include <hlir/ops/k210/fake_kpu_conv2d.h>
#include <hlir/ops/k210/kpu_conv2d.h>
#include <hlir/ops/k210/kpu_data_exchange.h>
#include <hlir/ops/quantize.h>
#include <hlir/transforms/k210/kpu_conv2d.h>
#include <hlir/transforms/k210/piecewise_regression.h>
#include <hlir/visitor.h>
#include <kernels/neutral/neutral_kernels.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::hlir::transforms;
using namespace nncase::hlir::transforms::k210;

namespace
{
auto quantize_weights(quantizer &quantizer, fake_kpu_conv2d &conv)
{
    auto weights = conv.weights();
    xt::xtensor<uint8_t, 4> q_weights(conv.weights().shape());
    std::vector<float> scales(conv.output_channels());
    auto total_range = quantizer.fixup_range(quantizer.get_range(weights.begin(), weights.end()));
    for (size_t oc = 0; oc < conv.output_channels(); oc++)
    {
        auto w_ch = xt::view(weights, oc, xt::all());
        auto range = quantizer.fixup_range(quantizer.get_range(w_ch.begin(), w_ch.end()));

        auto s1 = total_range.max / range.max;
        auto s2 = total_range.min / range.min;
        auto s = (s1 < 0 || s2 < 0) ? std::max(s1, s2) : std::min(s1, s2);

        assert(s > 0);
        for (auto &v : w_ch)
            v *= s;
        scales[oc] = s;
    }

    total_range = quantizer.get_range(weights.begin(), weights.end());
    auto q_p = quantizer.get_quant_param(total_range, 8);

    auto out_it = q_weights.begin();
    for (auto w : weights)
        *out_it++ = (uint8_t)std::clamp((int32_t)std::round(w * q_p.scale + q_p.zero_point), 0, 255);
    return std::make_tuple(q_p, std::move(scales), std::move(q_weights));
}

auto quantize_act(quantizer &quantizer, float act_in_scale, const quant_param_t &yq_p, const quant_param_t &zq_p, value_range<float> activation, fused_unary *fu)
{
    const auto xq_low = -(1LL << 35);
    const auto xq_high = (1LL << 35) - 1;
    const auto xf_min = std::clamp((0 - yq_p.zero_point) / yq_p.scale, activation.min, activation.max);
    const auto xf_max = std::clamp((255 - yq_p.zero_point) / yq_p.scale, activation.min, activation.max);
    const auto zq_scale = act_in_scale / zq_p.scale;

    const auto samples_count = 1024;
    const auto sample_step = (xf_max - xf_min) / (samples_count - 1);
    std::array<float, samples_count> samples_x, samples_y;
    for (int32_t i = 0; i < samples_count; i++)
        samples_x[i] = samples_y[i] = xf_min + i * sample_step;

    // 1. Non-clamp activation
    if (fu)
    {
        std::stringstream ss;
        runtime::binary_writer bw(ss);
        runtime::nnil_builder builder(bw);

        fused_unary::compile_graph(fu->subgraph(), builder);
        auto buf = ss.str();
        std::vector<uint8_t> body(reinterpret_cast<uint8_t *>(buf.data()), reinterpret_cast<uint8_t *>(buf.data() + buf.size()));
        kernels::neutral::nnil_unary_method(samples_x.data(), samples_y.data(), samples_count, body);
    }

    // 2. Piecewise regression
    std::vector<point> sample_points(samples_count);
    for (size_t i = 0; i < sample_points.size(); i++)
        sample_points[i] = { samples_x[i], samples_y[i] };
    piecewise_regression pr(15);
    auto segs = pr.fit(sample_points);

    assert(segs.size() == 15);
    kpu_activation_table_t act;
    act[0] = { 0x800000000, 0, 0, 0 };

    size_t i;
    for (i = 1; i < 16; i++)
    {
        auto &src = segs[i - 1];
        auto &dest = act[i];

        auto x0 = src.start * act_in_scale;
        auto mul = quantizer.get_fixed_mul(src.slop / zq_scale, 16, 20, true);
        dest.start_x = (int64_t)std::round(x0);
        dest.mul = mul.rounded_mul();
        dest.shift = mul.shift;
        dest.add = std::clamp((int32_t)std::round(src.intercept * zq_p.scale + zq_p.zero_point), 0, 255);
    }

    return act;
}

auto quantize_bn(quantizer &quantizer, fake_kpu_conv2d &conv, float sa, const std::vector<float> w_scales, const quant_param_t &yq_p)
{
    std::vector<kpu_batchnorm_segment> bn(conv.output_channels());
    auto &bias = conv.bias();
    auto so = yq_p.scale / sa;
    auto bn_mul = quantizer.get_fixed_mul(so, 22, 255, true);
    auto bn_shift = std::min(bn_mul.shift, (int8_t)15);
    auto up_scale = bn_mul.shift - bn_shift;
    assert(up_scale >= 0);
    auto post_mul = bn_mul.rounded_mul() / bn_mul.mul * std::pow(2.f, up_scale);
    auto s_act_in = yq_p.scale * post_mul;

    for (size_t i = 0; i < bias.size(); i++)
    {
        auto b = bias[i];
        auto ch_so = so * post_mul / w_scales[i];
        auto ch_bn_mul = quantizer.get_fixed_mul(ch_so, 22, 15, true);
        bn[i] = {
            ch_bn_mul.rounded_mul(),
            ch_bn_mul.shift,
            (int32_t)std::round(b * s_act_in)
        };
    }

    return std::make_tuple(std::move(bn), s_act_in);
}
}

bool kpu_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (auto conv = node_cast<fake_kpu_conv2d>(node))
    {
        if (conv->input().connection()->attributes() & cnctr_attr_need_quantize
            && conv->output().attributes() & cnctr_attr_need_quantize)
        {
            context.inputs.emplace_back(&conv->input());
            context.matched_nodes.emplace_back(conv);

            if (auto fu = try_get_direct_child<fused_unary>(*conv))
            {
                if (fu->input().connection()->attributes() & cnctr_attr_need_quantize
                    && fu->output().attributes() & cnctr_attr_need_quantize)
                {
                    context.outputs.emplace_back(&fu->output());
                    context.matched_nodes.emplace_back(fu);
                    return true;
                }
            }

            context.outputs.emplace_back(&conv->output());
            return true;
        }
    }

    return false;
}

void kpu_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[0]);
    auto old_fu = context.matched_nodes.size() > 1 ? static_cast<fused_unary *>(context.matched_nodes[1]) : nullptr;

    auto iq_p = quantizer_.get_quant_param(quantizer_.get(output), 8);
    auto [wq_p, w_scales, q_weights] = quantize_weights(quantizer_, old_conv);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_conv.output()), 8);
    auto zq_p = quantizer_.get_quant_param(quantizer_.get(*context.outputs[0]), 8);
    auto sa = iq_p.scale * wq_p.scale;
    auto [bn, s_act_in] = quantize_bn(quantizer_, old_conv, sa, w_scales, yq_p);
    auto act = quantize_act(quantizer_, s_act_in, yq_p, zq_p, old_conv.fused_activation(), old_fu);
    auto filter = get_kpu_filter_size(old_conv.filter_type());

    auto q = context.graph.emplace<quantize>(output.shape(), iq_p);
    q->name(output.owner().name() + "/quantize");
    auto upload = context.graph.emplace<kpu_upload>(q->output().shape());
    upload->name(output.owner().name() + "/kpu_upload");
    auto conv = context.graph.emplace<kpu_conv2d>(false, upload->output().shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        std::move(q_weights), (uint8_t)iq_p.zero_point, -wq_p.zero_point, 0, -iq_p.zero_point, 0, (int64_t)filter * filter * wq_p.zero_point * iq_p.zero_point,
        std::move(bn), std::move(act));
    conv->name(old_conv.name());
    auto download = context.graph.emplace<kpu_download>(conv->kpu_output().shape());
    upload->name(old_conv.name() + "/kpu_download");
    auto deq = context.graph.emplace<dequantize>(download->output().shape(), zq_p);
    deq->name(old_conv.name() + "/dequantize");
    link(*context.outputs[0], deq->output(), &quantizer_);

    upload->input().connect(q->output());
    conv->input().connect(upload->output());
    download->input().connect(conv->kpu_output());
    deq->input().connect(download->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
