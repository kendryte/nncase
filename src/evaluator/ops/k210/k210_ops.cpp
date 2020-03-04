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
#include <kernels/k210/k210_kernels.h>
#include <llir/evaluator.h>
#include <llir/ops/k210/fake_kpu_conv2d.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::llir;
using namespace nncase::llir::k210;

namespace nncase
{
namespace llir
{
    void register_k210_evaluators()
    {
        register_evaluator(op_k210_fake_kpu_conv2d, [](llir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<fake_kpu_conv2d &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            auto in_shape = rnode.input().shape();
            shape_t conv_out_shape { in_shape[0], (size_t)rnode.output_channels(), in_shape[2], in_shape[3] };
            auto conv_out_fmap_size = xt::compute_size(conv_out_shape);

            auto conv_output_tmp = std::make_unique<float[]>(conv_out_fmap_size);
            auto batch = in_shape[0];
            auto in_size_per_batch = xt::compute_size(in_shape) / batch;
            auto conv_output_tmp_size_per_batch = conv_out_fmap_size / batch;
            auto out_size_per_batch = xt::compute_size(rnode.output().shape()) / batch;
            auto p_input = input.data();
            auto p_conv_ouput_tmp = conv_output_tmp.get();
            auto p_output = output.data();

#define KPU_CONV2D_IMPL(is_depthwise_val, filter_size_val)                                                                      \
    if (rnode.is_depthwise() == is_depthwise_val && runtime::k210::get_kpu_filter_size(rnode.filter_type()) == filter_size_val) \
    kernels::k210::fake_kpu_conv2d<is_depthwise_val, filter_size_val>(p_input, p_conv_ouput_tmp, rnode.weights().data(),        \
        rnode.bias().data(), in_shape[2], in_shape[3], in_shape[1], rnode.output_channels(), rnode.fused_activation())

            for (size_t n = 0; n < batch; n++)
            {
                KPU_CONV2D_IMPL(true, 1);
                else KPU_CONV2D_IMPL(true, 3);
                else KPU_CONV2D_IMPL(false, 1);
                else KPU_CONV2D_IMPL(false, 3);

                kernels::k210::kpu_pool2d(p_conv_ouput_tmp, p_output, in_shape[2], in_shape[3], rnode.output_channels(),
                    rnode.pool_type());

                p_input += in_size_per_batch;
                p_conv_ouput_tmp += conv_output_tmp_size_per_batch;
                p_output += out_size_per_batch;
            }
        });
    }
}
}
