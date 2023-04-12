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
#pragma once
#include <nncase/ir/node.h>
#include <nncase/ir/ops/k210/opcode.h>
#include <nncase/runtime/k210/runtime_types.h>

namespace nncase::ir::k210 {
struct kpu_conv2d_quant_args {
    int32_t arg_x;
    int32_t shift_x;
    int32_t arg_w;
    int32_t shift_w;
    int64_t arg_add;

    friend bool operator==(const kpu_conv2d_quant_args &lhs,
                           const kpu_conv2d_quant_args &rhs) = default;
    friend bool operator!=(const kpu_conv2d_quant_args &lhs,
                           const kpu_conv2d_quant_args &rhs) = default;
};

class NNCASE_MODULES_K210_API kpu_conv2d : public node {
  public:
    DEFINE_NODE_OPCODE(op_k210_kpu_conv2d);

    const input_connector &weights() const { return input_at(1); }

    input_connector &input() { return input_at(0); }
    input_connector &weights() { return input_at(1); }
    input_connector &batch_norm() { return input_at(2); }
    input_connector &activation() { return input_at(3); }
    output_connector &kpu_output() { return output_at(0); }
    output_connector &main_mem_output() { return output_at(1); }

    bool has_main_mem_output() const noexcept { return outputs().size() == 2; }
    bool is_depthwise() const noexcept { return is_depthwise_; }
    int32_t input_channels() const noexcept {
        return is_depthwise() ? (int32_t)weights().shape()[0]
                              : (int32_t)weights().shape()[1];
    }
    int32_t output_channels() const noexcept {
        return (int32_t)weights().shape()[0];
    }
    runtime::k210::kpu_filter_type_t filter_type() const noexcept {
        return filter_type_;
    }
    runtime::k210::kpu_pool_type_t pool_type() const noexcept {
        return pool_type_;
    }
    uint8_t pad_value() const noexcept { return pad_value_; }
    const kpu_conv2d_quant_args &quant_args() const noexcept {
        return quant_args_;
    }
    const std::vector<nncase::runtime::k210::kpu_batchnorm_segment>
    bn() const noexcept {
        return bn_;
    }
    const nncase::runtime::k210::kpu_activation_table_t act() const noexcept {
        return act_;
    }

    kpu_conv2d(bool has_main_mem_output, shape_t input_shape, bool is_depthwise,
               shape_t weights_shape,
               runtime::k210::kpu_filter_type_t filter_type,
               runtime::k210::kpu_pool_type_t pool_type, uint8_t pad_value,
               kpu_conv2d_quant_args quant_args,
               std::vector<nncase::runtime::k210::kpu_batchnorm_segment> bn,
               nncase::runtime::k210::kpu_activation_table_t act);

  protected:
    bool properties_equal(node &other) const override;

  private:
    bool is_depthwise_;
    runtime::k210::kpu_filter_type_t filter_type_;
    runtime::k210::kpu_pool_type_t pool_type_;
    uint8_t pad_value_;
    kpu_conv2d_quant_args quant_args_;
    std::vector<nncase::runtime::k210::kpu_batchnorm_segment> bn_;
    nncase::runtime::k210::kpu_activation_table_t act_;
};
} // namespace nncase::ir::k210
