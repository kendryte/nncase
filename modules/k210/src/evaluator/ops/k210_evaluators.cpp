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
#include <nncase/ir/evaluator.h>
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/ir/ops/k210/k210_evaluators.h>
#include <nncase/ir/ops/k210/kpu_conv2d.h>
#include <nncase/ir/ops/k210/kpu_data_exchange.h>
#include <nncase/ir/ops/k210/opcode.h>
#include <nncase/kernels/k210/k210_kernels.h>
#include <nncase/runtime/k210/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::schedule;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime;

void ir::k210::register_k210_evaluators() {
    register_evaluator(op_k210_kpu_upload,
                       [](ir::node &node, function_evaluate_context &context) {
                           auto &rnode = static_cast<kpu_upload &>(node);

                           auto input = context.memory_at(rnode.input());
                           auto output = context.memory_at(rnode.output());
                           auto input_mem = input.buffer().as_span<uint8_t>();
                           auto output_mem = output.buffer().as_span<uint8_t>();
                           auto in_shape = input.shape();
                           assert(in_shape.size() == 4);
                           nncase::runtime::k210::kpu_shape_t kpu_in_shape{
                               (uint32_t)in_shape[0], (uint32_t)in_shape[1],
                               (uint32_t)in_shape[2], (uint32_t)in_shape[3]};

                           [[maybe_unused]] auto ret =
                               kernels::k210::kpu_upload(input_mem.data(),
                                                         output_mem.data(),
                                                         kpu_in_shape, 0);
                           // kpu_upload(const uint8_t *src, uint8_t *dest,
                           // const runtime::k210::kpu_shape_t &in_shape,
                           // uint32_t dma_ch)
                       });

    register_evaluator(op_k210_kpu_download,
                       [](ir::node &node, function_evaluate_context &context) {
                           auto &rnode = static_cast<kpu_download &>(node);

                           auto input = context.memory_at(rnode.input());
                           auto output = context.memory_at(rnode.output());
                           auto input_mem = input.buffer().as_span<uint8_t>();
                           auto output_mem = output.buffer().as_span<uint8_t>();
                           auto in_shape = input.shape();
                           assert(in_shape.size() == 4);
                           nncase::runtime::k210::kpu_shape_t kpu_in_shape{
                               (uint32_t)in_shape[0], (uint32_t)in_shape[1],
                               (uint32_t)in_shape[2], (uint32_t)in_shape[3]};

                           [[maybe_unused]] auto ret =
                               kernels::k210::kpu_download(input_mem.data(),
                                                           output_mem.data(),
                                                           kpu_in_shape);
                       });

    register_evaluator(
        op_k210_fake_kpu_conv2d,
        [](ir::node &node, function_evaluate_context &context) {
            auto &rnode = static_cast<fake_kpu_conv2d &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at(rnode.input());
            auto weights = context.memory_at(rnode.weights());
            auto bias = context.memory_at(rnode.bias());
            auto output = context.memory_at(rnode.output());
            auto input_mem = input.buffer().as_span<float>();
            auto output_mem = output.buffer().as_span<float>();
            auto weights_mem = weights.buffer().as_span<float>();
            auto bias_mem = bias.buffer().as_span<float>();

            auto in_shape = input.shape();
            shape_t conv_out_shape{in_shape[0], (size_t)rnode.output_channels(),
                                   in_shape[2], in_shape[3]};
            auto conv_out_fmap_size = runtime::compute_size(conv_out_shape);

            auto conv_output_tmp =
                std::make_unique<float[]>(conv_out_fmap_size);
            auto batch = in_shape[0];
            auto in_size_per_batch = runtime::compute_size(in_shape) / batch;
            auto conv_output_tmp_size_per_batch = conv_out_fmap_size / batch;
            auto out_size_per_batch =
                runtime::compute_size(rnode.output().shape()) / batch;
            auto p_input = input_mem.data();
            auto p_conv_ouput_tmp = conv_output_tmp.get();
            auto p_output = output_mem.data();

#define FAKE_KPU_CONV2D_IMPL(is_depthwise_val, filter_size_val)                \
    if (rnode.is_depthwise() == is_depthwise_val &&                            \
        runtime::k210::get_kpu_filter_size(rnode.filter_type()) ==             \
            filter_size_val)                                                   \
    kernels::k210::fake_kpu_conv2d<is_depthwise_val, filter_size_val>(         \
        p_input, p_conv_ouput_tmp, weights_mem.data(), bias_mem.data(),        \
        in_shape[2], in_shape[3], in_shape[1], rnode.output_channels(),        \
        rnode.fused_activation())

            for (size_t n = 0; n < batch; n++) {
                FAKE_KPU_CONV2D_IMPL(true, 1);
                else FAKE_KPU_CONV2D_IMPL(true, 3);
                else FAKE_KPU_CONV2D_IMPL(false, 1);
                else FAKE_KPU_CONV2D_IMPL(false, 3);

                kernels::k210::kpu_pool2d(
                    p_conv_ouput_tmp, p_output, in_shape[2], in_shape[3],
                    rnode.output_channels(), rnode.pool_type());

                p_input += in_size_per_batch;
                p_conv_ouput_tmp += conv_output_tmp_size_per_batch;
                p_output += out_size_per_batch;
            }
        });

    register_evaluator(
        op_k210_kpu_conv2d,
        [](ir::node &node, function_evaluate_context &context) {
            auto &rnode = static_cast<kpu_conv2d &>(node);

            auto input = context.memory_at(rnode.input());
            auto weights = context.memory_at(rnode.weights());
            auto batch_norm = context.memory_at(rnode.batch_norm());
            auto activation = context.memory_at(rnode.activation());
            auto output = context.memory_at(rnode.kpu_output());

            auto input_mem = input.buffer().as_span<uint8_t>();
            auto output_mem = output.buffer().as_span<uint8_t>();
            auto weights_mem = weights.buffer().as_span<uint8_t>();
            [[maybe_unused]] auto batch_norm_mem =
                batch_norm.buffer().as_span<uint64_t>();
            [[maybe_unused]] auto activation_mem =
                activation.buffer().as_span<uint8_t>();

            auto in_shape = input.shape();
            auto out_shape = output.shape();
            auto pad_value = rnode.pad_value();
            auto quant_args = rnode.quant_args();
            auto bn = rnode.bn();
            auto act = rnode.act();
            shape_t conv_out_shape{in_shape[0], (size_t)rnode.output_channels(),
                                   in_shape[2], in_shape[3]};
            auto conv_out_fmap_size = runtime::compute_size(conv_out_shape);
            nncase::runtime::k210::kpu_shape_t kpu_in_shape{
                (uint32_t)in_shape[0], (uint32_t)in_shape[1],
                (uint32_t)in_shape[2], (uint32_t)in_shape[3]};
            nncase::runtime::k210::kpu_shape_t kpu_out_shape{
                (uint32_t)in_shape[0], (uint32_t)rnode.output_channels(),
                (uint32_t)out_shape[2], (uint32_t)out_shape[3]};

            auto conv_output_tmp =
                std::make_unique<uint8_t[]>(conv_out_fmap_size);
            auto batch = in_shape[0];
            auto in_size_per_batch = runtime::compute_size(in_shape) / batch;
            auto conv_output_tmp_size_per_batch = conv_out_fmap_size / batch;
            auto out_size_per_batch =
                runtime::compute_size(rnode.outputs()[0]->shape()) / batch;
            auto p_input = input_mem.data();
            auto p_conv_ouput_tmp = conv_output_tmp.get();
            auto p_output = output_mem.data();
            auto download_output_tmp =
                std::make_unique<uint8_t[]>(runtime::compute_size(in_shape));
            auto p_download_output_tmp = download_output_tmp.get();
            auto pool_output_tmp =
                std::make_unique<uint8_t[]>(runtime::compute_size(out_shape));
            auto p_pool_output_tmp = pool_output_tmp.get();

            const auto groups =
                rnode.is_depthwise() ? rnode.output_channels() : 1;
            const auto g_oc =
                rnode.is_depthwise() ? 1 : rnode.output_channels();

            [[maybe_unused]] auto ret_dl = kernels::k210::kpu_download(
                p_input, p_download_output_tmp, kpu_in_shape);

#define KPU_CONV2D_IMPL(is_depthwise_val, filter_size_val)                     \
    if (rnode.is_depthwise() == is_depthwise_val &&                            \
        runtime::k210::get_kpu_filter_size(rnode.filter_type()) ==             \
            filter_size_val) {                                                 \
        kernels::k210::kpu_conv2d<is_depthwise_val, filter_size_val>(          \
            p_download_output_tmp, workspace.get(), p_conv_ouput_tmp,          \
            weights_mem.data(), in_shape[2], in_shape[3], in_shape[1],         \
            rnode.output_channels(), pad_value, quant_args.arg_x,              \
            quant_args.shift_x, quant_args.arg_w, quant_args.shift_w,          \
            quant_args.arg_add, &bn[0], act);                                  \
    }

            auto workspace = std::make_unique<int64_t[]>(
                groups * g_oc * in_shape[2] * in_shape[3]);
            auto p_pool_output_tmp_base = p_pool_output_tmp;
            for (size_t n = 0; n < batch; n++) {
                KPU_CONV2D_IMPL(true, 1)
                else KPU_CONV2D_IMPL(true, 3) else KPU_CONV2D_IMPL(
                    false, 1) else KPU_CONV2D_IMPL(false, 3)

                    kernels::k210::kpu_pool2d(
                        p_conv_ouput_tmp, p_pool_output_tmp, in_shape[2],
                        in_shape[3], rnode.output_channels(),
                        rnode.pool_type());

                p_input += in_size_per_batch;
                p_conv_ouput_tmp += conv_output_tmp_size_per_batch;
                p_pool_output_tmp += out_size_per_batch;
            }

            [[maybe_unused]] auto ret_ul = kernels::k210::kpu_upload(
                p_pool_output_tmp_base, p_output, kpu_out_shape, 0);
        });
}
