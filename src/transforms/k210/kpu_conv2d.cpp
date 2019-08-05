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

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

namespace
{
auto quantize_weights(quantizer &quantizer, fake_kpu_conv2d &conv)
{
    auto &weights = conv.weights();
    xt::xtensor<uint8_t, 4> q_weights(conv.weights().shape());
    auto q_p = quantizer.get_quant_param(quantizer.get_range(weights.begin(), weights.end()), 8);

    auto out_it = q_weights.begin();
    for (auto w : weights)
        *out_it++ = (uint8_t)std::clamp((int32_t)std::round(w * q_p.scale + q_p.zero_point), 0, 255);
    return std::make_tuple(q_p, std::move(q_weights));
}

auto quantize_act(quantizer &quantizer, float post_mul)
{
    kpu_activation_table_t act;
    const std::array<int64_t, 16> starts {
        0x800000000, 0xf7d4cf4b8, 0xf8ed5a20c, 0xfa05e4f60,
        0xfb2e05baa, 0xfc46908fe, 0xfd5f1b652, 0xfe77a63a6,
        0xff9fc6ff0, 0xfffd4a9b7, 0, 0x7FFFFFFF0,
        0x7FFFFFFF1, 0x7FFFFFFF2, 0x7FFFFFFF3, 0x7FFFFFFF4
    };

    for (size_t i = 0; i < starts.size(); i++)
    {
        auto &seg = act[i];
        seg.start_x = starts[i];

        if (i == 10)
        {
            auto mul = quantizer.get_fixed_mul(1.f / post_mul, 16, 20, true);
            seg.mul = mul.rounded_mul();
            seg.shift = mul.shift;
            seg.add = 0;
        }
        else
        {
            seg.mul = 0;
            seg.shift = 0;
            seg.add = 0;
        }
    }

    return act;
}

auto quantize_bn_act(quantizer &quantizer, fake_kpu_conv2d &conv, float sa, const quant_param_t &yq_p)
{
    std::vector<kpu_batchnorm_segment> bn(conv.output_channels());
    auto &bias = conv.bias();
    auto so = yq_p.scale / sa;
    auto bn_mul = quantizer.get_fixed_mul(so, 22, 255, true);
    auto bn_shift = std::min(bn_mul.shift, (int8_t)15);
    auto up_scale = bn_mul.shift - bn_shift;
    assert(up_scale > 0);
    auto post_mul = bn_mul.rounded_mul() / bn_mul.mul * std::pow(2, up_scale);

    for (size_t i = 0; i < bias.size(); i++)
    {
        auto b = bias[i];
        bn[i] = {
            bn_mul.rounded_mul(),
            bn_shift,
            (int32_t)std::round((b * yq_p.scale + yq_p.zero_point) * post_mul)
        };
    }

    return std::make_tuple(std::move(bn), quantize_act(quantizer, post_mul));
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
    auto [wq_p, q_weights] = quantize_weights(quantizer_, old_conv);
    auto yq_p = quantizer_.get_quant_param(quantizer_.get(old_deq.output()), 8);
    auto sa = iq_p.scale * wq_p.scale;
    auto [bn, act] = quantize_bn_act(quantizer_, old_conv, sa, yq_p);
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
