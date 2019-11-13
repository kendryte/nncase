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
#include <algorithm>
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <ir/ops/k210/kpu_conv2d.h>
#include <ir/ops/k210/kpu_data_exchange.h>
#include <ir/ops/quantize.h>
#include <ir/visitor.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <transforms/k210/kpu_conv2d.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

namespace
{
struct line
{
    float start_x;
    float end_x;
    float k;
    float b;
};

xt::svector<line> to_lines(const xt::svector<piecewise_linear_segment> &segs)
{
    xt::svector<line> lines;
    for (size_t i = 0; i < segs.size(); i++)
    {
        auto end = i == segs.size() - 1 ? std::numeric_limits<float>::max() : segs[i + 1].start;
        lines.push_back({ segs[i].start, end, segs[i].mul, segs[i].add });
    }

    return lines;
}

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

xt::svector<piecewise_linear_segment> clamp_act(const quant_param_t &yq_p, const xt::svector<piecewise_linear_segment> &activation)
{
    auto y_start = (0 - yq_p.zero_point) / yq_p.scale;
    auto y_end = (255 - yq_p.zero_point) / yq_p.scale;
    auto lines = to_lines(activation);

    xt::svector<piecewise_linear_segment> segs;
    for (auto &&line : lines)
    {
        if (line.k)
        {
            auto x_start = (y_start - line.b) / line.k;
            auto x_end = (y_end - line.b) / line.k;
            auto [x_min, x_max] = std::minmax(x_start, x_end);
            auto r_x_min = std::max(x_min, line.start_x);
            auto r_x_max = std::min(x_max, line.end_x);
            if (r_x_min < r_x_max)
                segs.push_back({ r_x_min, line.k, line.b });
        }
    }

    std::sort(segs.begin(), segs.end(), [](auto &a, auto &b) { return a.start < b.start; });
    return segs;
}

auto quantize_act(quantizer &quantizer, float post_mul, const quant_param_t &yq_p, const xt::svector<piecewise_linear_segment> &activation)
{
    auto segs = clamp_act(yq_p, activation);
    assert(segs.size() < 16);
    kpu_activation_table_t act;
    act[0] = { 0x800000000, 0, 0, 0 };

    size_t i;
    for (i = 0; i < segs.size(); i++)
    {
        auto &src = segs[i];
        auto &dest = act[i + 1];

        auto x0 = src.start * yq_p.scale * post_mul;
        auto mul = quantizer.get_fixed_mul(src.mul / post_mul, 16, 20, true);
        auto start_value = src.start * src.mul + src.add;
        dest.start_x = (int64_t)std::round(x0);
        dest.mul = mul.rounded_mul();
        dest.shift = mul.shift;
        dest.add = std::clamp((int32_t)std::round(start_value * yq_p.scale + yq_p.zero_point), 0, 255);
    }

    // dummy
    auto last_seg = segs.back();
    for (; i < 15; i++)
    {
        auto &dest = act[i + 1];
        dest.start_x = 0x7FFFFFFF0 + i;
        auto last_value = (dest.start_x / (yq_p.scale * post_mul)) * last_seg.mul + last_seg.add;
        dest.mul = 0;
        dest.shift = 0;
        dest.add = std::clamp((int32_t)std::round(last_value * yq_p.scale + yq_p.zero_point), 0, 255);
    }

    return act;
}

auto quantize_bn_act(quantizer &quantizer, fake_kpu_conv2d &conv, float sa, const std::vector<float> w_scales, const quant_param_t &yq_p, const xt::svector<piecewise_linear_segment> &activation)
{
    std::vector<kpu_batchnorm_segment> bn(conv.output_channels());
    auto &bias = conv.bias();
    auto so = yq_p.scale / sa;
    auto bn_mul = quantizer.get_fixed_mul(so, 22, 255, true);
    auto bn_shift = std::min(bn_mul.shift, (int8_t)15);
    auto up_scale = bn_mul.shift - bn_shift;
    assert(up_scale >= 0);
    auto post_mul = bn_mul.rounded_mul() / bn_mul.mul * std::pow(2.f, up_scale);

    for (size_t i = 0; i < bias.size(); i++)
    {
        auto b = bias[i];
        auto ch_so = so * post_mul / w_scales[i];
        auto ch_bn_mul = quantizer.get_fixed_mul(ch_so, 22, 15, true);
        bn[i] = {
            ch_bn_mul.rounded_mul(),
            ch_bn_mul.shift,
            (int32_t)std::round(b * yq_p.scale * post_mul)
        };
    }

    return std::make_tuple(std::move(bn), quantize_act(quantizer, post_mul, yq_p, activation));
}
}

bool kpu_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_fake_quantize)
    {
        auto &q = static_cast<fake_quantize &>(node);
        if (auto conv = try_get_direct_child<fake_kpu_conv2d>(q))
        {
            if (auto deq = try_get_direct_child<fake_dequantize>(*conv))
            {
                context.inputs.emplace_back(&q.input());
                context.outputs.emplace_back(&deq->output());

                context.matched_nodes.emplace_back(&q);
                context.matched_nodes.emplace_back(conv);
                context.matched_nodes.emplace_back(deq);
                return true;
            }
        }
    }

    return false;
}

void kpu_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_q = static_cast<fake_quantize &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[1]);
    auto &old_deq = static_cast<fake_dequantize &>(*context.matched_nodes[2]);

    auto iq_p = quantizer_.get_quant_param(quantizer_.get(old_q.output()), 8);
    auto [wq_p, w_scales, q_weights] = quantize_weights(quantizer_, old_conv);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_deq.output()), 8);
    auto sa = iq_p.scale * wq_p.scale;
    auto [bn, act] = quantize_bn_act(quantizer_, old_conv, sa, w_scales, yq_p, old_conv.fused_activation());
    auto filter = get_kpu_filter_size(old_conv.filter_type());

    auto q = context.graph.emplace<quantize>(output.shape(), iq_p);
    auto upload = context.graph.emplace<kpu_upload>(q->output().shape());
    auto conv = context.graph.emplace<kpu_conv2d>(false, upload->output().shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        std::move(q_weights), (uint8_t)iq_p.zero_point, -wq_p.zero_point, 0, -iq_p.zero_point, 0, (int64_t)filter * filter * wq_p.zero_point * iq_p.zero_point,
        std::move(bn), std::move(act));
    auto download = context.graph.emplace<kpu_download>(conv->kpu_output().shape());
    auto deq = context.graph.emplace<dequantize>(download->output().shape(), yq_p);
    upload->input().connect(q->output());
    conv->input().connect(upload->output());
    download->input().connect(conv->kpu_output());
    deq->input().connect(download->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
