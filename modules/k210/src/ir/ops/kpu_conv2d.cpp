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
#include <nncase/ir/ops/k210/kpu_conv2d.h>
#include <nncase/runtime/k210/runtime_module.h>
#include <nncase/runtime/k210/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

kpu_conv2d::kpu_conv2d(bool has_main_mem_output, shape_t input_shape,
                       bool is_depthwise, shape_t weights_shape,
                       runtime::k210::kpu_filter_type_t filter_type,
                       runtime::k210::kpu_pool_type_t pool_type,
                       uint8_t pad_value, kpu_conv2d_quant_args quant_args,
                       std::vector<kpu_batchnorm_segment> bn,
                       kpu_activation_table_t act)
    : is_depthwise_(is_depthwise),
      filter_type_(filter_type),
      pool_type_(pool_type),
      pad_value_(pad_value),
      quant_args_(quant_args),
      bn_(bn),
      act_(act) {
    module_type(k210_module_type);
    add_input("input", dt_uint8, input_shape);
    add_input("weights", dt_uint8, weights_shape);
    add_input("batch_norm", dt_uint64, shape_t{(size_t)output_channels()});
    add_input("activation", dt_uint8, shape_t{sizeof(kpu_activate_table_t)});
    add_output("output", dt_uint8,
               shape_t{input_shape[0], (size_t)output_channels(),
                       (size_t)get_kpu_pool_output_size((int32_t)input_shape[2],
                                                        pool_type_),
                       (size_t)get_kpu_pool_output_size((int32_t)input_shape[3],
                                                        pool_type_)},
               mem_kpu)
        .attributes(cnctr_attr_no_layout_strides);

    if (has_main_mem_output)
        add_output("main_mem_output", dt_uint8, kpu_output().shape(), mem_data)
            .attributes(cnctr_attr_no_layout_strides);
}

bool kpu_conv2d::properties_equal(node &other) const {
    auto &r = static_cast<kpu_conv2d &>(other);
    return is_depthwise() == r.is_depthwise() &&
           filter_type() == r.filter_type() && pool_type() == r.pool_type() &&
           pad_value() == r.pad_value() && quant_args() == r.quant_args() &&
           bn() == r.bn() && act() == r.act();
}
