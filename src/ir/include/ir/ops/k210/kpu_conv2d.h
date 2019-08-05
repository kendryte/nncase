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
#pragma once
#include "../../node.h"
#include <runtime/k210/k210_sim_types.h>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    namespace k210
    {
        class kpu_conv2d : public node
        {
        public:
            DEFINE_NODE_OPCODE(op_k210_kpu_conv2d);

            input_connector &input() { return input_at(0); }
            output_connector &kpu_output() { return output_at(0); }
            output_connector &main_mem_output() { return output_at(1); }

            bool has_main_mem_output() const noexcept { return outputs().size() == 2; }
            bool is_depthwise() const noexcept { return is_depthwise_; }
            int32_t input_channels() const noexcept { return is_depthwise() ? (int32_t)weights_.shape()[0] : (int32_t)weights_.shape()[1]; }
            int32_t output_channels() const noexcept { return (int32_t)weights_.shape()[0]; }
            runtime::k210::kpu_filter_type_t filter_type() const noexcept { return filter_type_; }
            runtime::k210::kpu_pool_type_t pool_type() const noexcept { return pool_type_; }
            const xt::xtensor<uint8_t, 4> &weights() const noexcept { return weights_; }
            uint8_t pad_value() const noexcept { return pad_value_; }
            int32_t arg_x() const noexcept { return arg_x_; }
            int32_t shift_x() const noexcept { return shift_x_; }
            int32_t arg_w() const noexcept { return arg_w_; }
            int32_t shift_w() const noexcept { return shift_w_; }
            int64_t arg_add() const noexcept { return arg_add_; }
            const std::vector<runtime::k210::kpu_batchnorm_segment> &batch_norm() const noexcept { return batch_norm_; }
            const runtime::k210::kpu_activation_table_t activation() const noexcept { return activation_; }

            kpu_conv2d(bool has_main_mem_output, shape_t input_shape, bool is_depthwise, runtime::k210::kpu_filter_type_t filter_type, runtime::k210::kpu_pool_type_t pool_type, xt::xtensor<uint8_t, 4> weights,
                uint8_t pad_value, int32_t arg_x, int32_t shift_x, int32_t arg_w, int32_t shift_w, int64_t arg_add, std::vector<runtime::k210::kpu_batchnorm_segment> batch_norm, runtime::k210::kpu_activation_table_t activation);

        private:
            xt::xtensor<uint8_t, 4> weights_;
            bool is_depthwise_;
            runtime::k210::kpu_filter_type_t filter_type_;
            runtime::k210::kpu_pool_type_t pool_type_;
            uint8_t pad_value_;
            int32_t arg_x_;
            int32_t shift_x_;
            int32_t arg_w_;
            int32_t shift_w_;
            int64_t arg_add_;
            std::vector<runtime::k210::kpu_batchnorm_segment> batch_norm_;
            runtime::k210::kpu_activation_table_t activation_;
        };
    }
}
}
