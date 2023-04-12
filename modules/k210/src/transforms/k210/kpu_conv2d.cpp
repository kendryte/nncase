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
#include <algorithm>
#include <nncase/codegen/nnil_builder.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/fused_unary.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/k210/kpu_conv2d.h>
#include <nncase/ir/ops/k210/kpu_data_exchange.h>
#include <nncase/ir/ops/k210/runtime_type_utils.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/visitor.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/nnil.h>
#include <nncase/runtime/k210/runtime_op_utility.h>
#include <nncase/transforms/k210/kpu_conv2d.h>
#include <nncase/transforms/k210/piecewise_regression.h>
#include <xtensor/xview.hpp>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::ir::transforms;
using namespace nncase::ir::transforms::k210;

namespace {
constexpr int32_t w_zero_point = 128;

auto quantize_weights(quantizer &quantizer, fake_kpu_conv2d &conv,
                      constant &weights, bool use_mse_quant_w) {
    auto old_weights_data = as_span<const float>(weights.data());
    std::vector<float> weights_data(old_weights_data.begin(),
                                    old_weights_data.end());
    std::vector<uint8_t> q_weights(xt::compute_size(conv.weights().shape()));
    std::vector<float> scales(conv.output_channels());
    const auto channel_w_size = q_weights.size() / conv.output_channels();

    if (use_mse_quant_w) {
        for (size_t oc = 0; oc < (size_t)conv.output_channels(); oc++) {
            std::span<float> w_ch(weights_data.data() + oc * channel_w_size,
                                  channel_w_size);
            std::span<uint8_t> qw_ch(q_weights.data() + oc * channel_w_size,
                                     channel_w_size);
            std::vector<float> deqw_ch(channel_w_size);
            auto range = quantizer.fixup_range(
                quantizer.get_range(w_ch.begin(), w_ch.end()), true);
            uint32_t steps_num = 256;
            float step = (range.max - range.min) / steps_num;
            float min_max_step = 0.f;
            float min_mse = FLT_MAX;
            std::vector<uint8_t> q_weights_channel(channel_w_size);
            std::vector<float> deq_weights_channel(channel_w_size);

            // choose the best min_max_step (min_step and max_step are same for
            // symmetric quantize)
            for (uint32_t i = 0; i < steps_num / 8; i++) {
                value_range<float> new_range{range.min - step * i,
                                             range.max + step * i};
                auto q_p = quantizer.get_quant_param(
                    new_range, 8, quantizer::quant_mode::unsigned_mode);
                for (size_t i = 0; i < w_ch.size(); i++) {
                    qw_ch[i] = kernels::detail::quantize<uint8_t>(w_ch[i], q_p);
                    deqw_ch[i] = (qw_ch[i] - q_p.zero_point) * q_p.scale;
                }

                float mse_i = 0.f;
                for (uint32_t j = 0; j < qw_ch.size(); j++) {
                    mse_i += std::pow((w_ch[j] - deqw_ch[j]), 2);
                }

                if (min_mse > mse_i) {
                    min_max_step = step * i;
                    min_mse = mse_i;
                }
            }

            // quant with range refined with min_max_step
            value_range<float> final_range{range.min - min_max_step,
                                           range.max + min_max_step};
            auto q_p = quantizer.get_quant_param(
                final_range, 8, quantizer::quant_mode::unsigned_mode);
            for (size_t i = 0; i < w_ch.size(); i++)
                qw_ch[i] = kernels::detail::quantize<uint8_t>(w_ch[i], q_p);
            scales[oc] = q_p.scale;
        }

        return std::make_tuple(std::move(scales), std::move(q_weights));
    } else {
        for (size_t oc = 0; oc < (size_t)conv.output_channels(); oc++) {
            std::span<float> w_ch(weights_data.data() + oc * channel_w_size,
                                  channel_w_size);
            std::span<uint8_t> qw_ch(q_weights.data() + oc * channel_w_size,
                                     channel_w_size);
            auto range = quantizer.fixup_range(
                quantizer.get_range(w_ch.begin(), w_ch.end()), true);
            auto q_p = quantizer.get_quant_param(
                range, 8, quantizer::quant_mode::unsigned_mode);
            for (size_t i = 0; i < w_ch.size(); i++)
                qw_ch[i] = kernels::detail::quantize<uint8_t>(w_ch[i], q_p);
            scales[oc] = q_p.scale;
        }

        return std::make_tuple(std::move(scales), std::move(q_weights));
    }
}

auto quantize_act(quantizer &quantizer, float act_in_scale,
                  const quant_param_t &yq_p, const quant_param_t &zq_p,
                  value_range<float> activation, fused_unary *fu) {
    const auto xf_min = std::clamp((0 - yq_p.zero_point) * yq_p.scale,
                                   activation.min, activation.max);
    const auto xf_max = std::clamp((255 - yq_p.zero_point) * yq_p.scale,
                                   activation.min, activation.max);
    const auto zq_scale = act_in_scale / zq_p.scale;

    const size_t samples_count = 2048;
    const auto sample_step = (xf_max - xf_min) / (samples_count - 1);
    std::array<float, samples_count> samples_x, samples_y;
    for (size_t i = 0; i < samples_count; i++)
        samples_x[i] = samples_y[i] = xf_min + i * sample_step;

    // 1. Non-clamp activation
    if (fu) {
        std::stringstream ss;
        binary_writer bw(ss);
        nnil_builder builder(bw);

        fused_unary::compile_graph(fu->subgraph(), builder);
        auto buf = ss.str();
        std::vector<gsl::byte> body(
            reinterpret_cast<gsl::byte *>(buf.data()),
            reinterpret_cast<gsl::byte *>(buf.data() + buf.size()));
        kernels::nnil_unary_method(samples_x.data(), samples_y.data(),
                                   samples_count, body)
            .unwrap_or_throw();
    }

    // 2. Piecewise regression
    std::vector<point> sample_points(samples_count);
    for (size_t i = 0; i < sample_points.size(); i++)
        sample_points[i] = {samples_x[i], samples_y[i]};
    piecewise_regression pr(15);
    auto segs = pr.fit(sample_points);

    assert(segs.size() == 15);
    kpu_activation_table_t act;
    act[0] = {0x800000000, 0, 0, 0};

    size_t i;
    for (i = 1; i < 16; i++) {
        auto &src = segs[i - 1];
        auto &dest = act[i];

        auto x0 = src.start / act_in_scale;
        auto mul = quantizer.get_fixed_mul(src.slop * zq_scale, 16, 43, true);
        dest.start_x = (int64_t)std::llrint(x0);
        dest.mul = mul.rounded_mul();
        dest.shift = mul.shift;
        dest.add = std::clamp(
            (int32_t)std::lrint(src.intercept / zq_p.scale + zq_p.zero_point),
            0, 255);
    }

    return act;
}

auto quantize_bn(quantizer &quantizer, fake_kpu_conv2d &conv, constant &bias,
                 float sx, const std::vector<float> w_scales,
                 [[maybe_unused]] const quant_param_t &yq_p) {
    auto bias_data = as_span<const float>(bias.data());
    std::vector<kpu_batchnorm_segment> bn(conv.output_channels());
    const auto filter = get_kpu_filter_size(conv.filter_type());
    const auto x_bits = 9 + 9 - 1 +
                        (uint32_t)std::ceil(std::log2(
                            (conv.is_depthwise() ? 1 : conv.input_channels()) +
                            filter * filter));
    const auto max_sw = *std::max_element(w_scales.begin(), w_scales.end());
    const auto minmax_b =
        std::minmax_element(bias_data.begin(), bias_data.end());
    const auto max_b =
        std::max(std::abs(*minmax_b.first), std::abs(*minmax_b.second));
    const auto max_scaled_b = (1LL << 31) - 1.0;
    const auto max_sa = sx * max_sw;
    size_t max_so_bits = 21; // BN_MUL_BITS restriction
    max_so_bits = std::min(
        max_so_bits, KPU_BN_OUT_BITS - x_bits); // KPU_BN_OUT_BITS restriction
    const auto max_so = (1LL << max_so_bits) - 1.0;
    auto min_s_act_in = max_sa / max_so;
    min_s_act_in = std::max(min_s_act_in, max_b / max_scaled_b);

    // Search closest s_act_in
    double s_act_in = yq_p.scale;
    while (true) {
        auto new_s_act_in = s_act_in / 2.0;
        if (new_s_act_in < min_s_act_in)
            break;
        else
            s_act_in = new_s_act_in;
    }

    for (size_t i = 0; i < bias_data.size(); i++) {
        auto sa = sx * w_scales[i];
        auto so = sa / s_act_in;
        auto bn_mul = quantizer.get_fixed_mul((float)so, 22, 15, true);
        auto b = bias_data[i];
        bn[i] = {bn_mul.rounded_mul(), bn_mul.shift,
                 (int32_t)std::lrint(b / s_act_in)};
    }

    return std::make_tuple(std::move(bn), s_act_in);
}

std::vector<kpu_batchnorm_argument_t>
to_runtime(const std::vector<kpu_batchnorm_segment> &batch_norm) {
    std::vector<kpu_batchnorm_argument_t> result(batch_norm.size());
    for (size_t i = 0; i < result.size(); i++) {
        auto &src = batch_norm[i];
        auto &dest = result[i];

        dest.batchnorm.data.norm_add = (uint32_t)src.add;
        dest.batchnorm.data.norm_mul = (uint32_t)src.mul;
        dest.batchnorm.data.norm_shift = (uint8_t)src.shift;
    }

    return result;
}

kpu_activate_table_t to_runtime(const kpu_activation_table_t &act) {
    kpu_activate_table_t result;
    for (size_t i = 0; i < 16; i++) {
        auto &src = act[i];
        auto &dest = result.activate_para[i];
        auto &bias = i < 8 ? result.activate_para_bias0.data.result_bias[i]
                           : result.activate_para_bias1.data.result_bias[i - 8];

        dest.data.x_start = (uint64_t)src.start_x;
        dest.data.y_mul = (uint16_t)src.mul;
        dest.data.shift_number = (uint8_t)src.shift;
        bias = (uint8_t)src.add;
    }

    return result;
}
} // namespace

bool kpu_conv2d_transform::on_try_match(node &node,
                                        transform_context &context) {
    fake_kpu_conv2d *conv;
    constant *weights, *bias;
    if ((conv = node_cast<fake_kpu_conv2d>(node)) &&
        (weights = try_get_direct_parent<constant>(*conv, 1)) &&
        (bias = try_get_direct_parent<constant>(*conv, 2))) {
        if (conv->input().connection()->attributes() &
                cnctr_attr_need_quantize &&
            conv->output().attributes() & cnctr_attr_need_quantize) {
            context.inputs.emplace_back(&conv->input());
            context.matched_nodes.emplace_back(conv);
            context.matched_nodes.emplace_back(weights);
            context.matched_nodes.emplace_back(bias);

            if (auto fu = try_get_direct_child<fused_unary>(*conv)) {
                if (fu->input().connection()->attributes() &
                        cnctr_attr_need_quantize &&
                    fu->output().attributes() & cnctr_attr_need_quantize) {
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

void kpu_conv2d_transform::process(transform_context &context) {
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[0]);
    auto &weights = static_cast<constant &>(*context.matched_nodes[1]);
    auto &bias = static_cast<constant &>(*context.matched_nodes[2]);
    auto old_fu = context.matched_nodes.size() > 3
                      ? static_cast<fused_unary *>(context.matched_nodes[3])
                      : nullptr;

    auto &quantizer = *context.quantizer;
    auto iq_p = quantizer.get_quant_param(quantizer.get(output), 8,
                                          quantizer::quant_mode::unsigned_mode);
    auto [w_scales, q_weights] =
        quantize_weights(quantizer, old_conv, weights, use_mse_quant_w_);
    auto yq_p = quantizer.get_quant_param(quantizer.get(old_conv.output()), 8,
                                          quantizer::quant_mode::unsigned_mode);
    auto zq_p = quantizer.get_quant_param(quantizer.get(*context.outputs[0]), 8,
                                          quantizer::quant_mode::unsigned_mode);
    auto [bn, s_act_in] =
        quantize_bn(quantizer, old_conv, bias, iq_p.scale, w_scales, yq_p);
    auto act = quantize_act(quantizer, (float)s_act_in, yq_p, zq_p,
                            old_conv.fused_activation(), old_fu);
    auto filter = get_kpu_filter_size(old_conv.filter_type());
    auto rt_bn = to_runtime(bn);
    auto rt_act = to_runtime(act);

    auto q = context.graph.emplace<quantize>(dt_float32, output.shape(),
                                             dt_uint8, iq_p);
    q->name(output.owner().name() + "/quantize");
    auto upload = context.graph.emplace<kpu_upload>(q->output().shape());
    upload->name(output.owner().name() + "/kpu_upload");

    auto c_weights = context.graph.emplace<constant>(
        dt_uint8, old_conv.weights().shape(), q_weights);
    c_weights->name(weights.name());
    auto c_bn = context.graph.emplace<constant>(
        dt_uint64, shape_t{(size_t)old_conv.output_channels()}, rt_bn);
    c_bn->name(bias.name());
    auto c_act = context.graph.emplace<constant>(
        dt_uint8, shape_t{sizeof(kpu_activate_table_t)},
        gsl::make_span(&rt_act, 1));
    c_act->name(old_conv.name() + "/act");
    c_act->alignment(256);
    kpu_conv2d_quant_args q_args{};
    q_args.arg_x = (int32_t)-w_zero_point;
    q_args.shift_x = 0;
    q_args.arg_w = (int32_t)-iq_p.zero_point;
    q_args.shift_w = 0;
    q_args.arg_add = (int64_t)filter * filter * w_zero_point * iq_p.zero_point;

    auto conv = context.graph.emplace<kpu_conv2d>(
        false, upload->output().shape(), old_conv.is_depthwise(),
        old_conv.weights().shape(), old_conv.filter_type(),
        old_conv.pool_type(), (uint8_t)iq_p.zero_point, q_args, bn, act);
    conv->name(old_conv.name());
    conv->weights().connect(c_weights->output());
    conv->batch_norm().connect(c_bn->output());
    conv->activation().connect(c_act->output());

    auto download =
        context.graph.emplace<kpu_download>(conv->kpu_output().shape());
    upload->name(old_conv.name() + "/kpu_download");
    auto deq = context.graph.emplace<dequantize>(download->output().type(),
                                                 download->output().shape(),
                                                 dt_float32, zq_p);
    deq->record_output_connectors_quant_map(deq->output_at(0),
                                            old_conv.output_at(0));
    deq->record_node_name_before_quant(old_conv.name());
    deq->name(old_conv.name() + "/dequantize");
    link(*context.outputs[0], deq->output(), &quantizer);

    upload->input().connect(q->output());
    conv->input().connect(upload->output());
    download->input().connect(conv->kpu_output());
    deq->input().connect(download->output());

    q->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
