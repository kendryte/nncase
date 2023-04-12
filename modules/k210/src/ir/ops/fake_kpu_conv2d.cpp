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
#include <nncase/ir/ops/k210/fake_kpu_conv2d.h>
#include <nncase/runtime/k210/runtime_module.h>
#include <nncase/runtime/k210/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

fake_kpu_conv2d::fake_kpu_conv2d(shape_t input_shape, bool is_depthwise,
                                 shape_t weights_shape,
                                 runtime::k210::kpu_filter_type_t filter_type,
                                 runtime::k210::kpu_pool_type_t pool_type,
                                 value_range<float> fused_activation)
    : is_depthwise_(is_depthwise),
      filter_type_(filter_type),
      pool_type_(pool_type),
      fused_activation_(fused_activation) {
    module_type(k210_module_type);
    add_input("input", dt_float32, input_shape);
    add_input("weights", dt_float32, weights_shape);
    add_input("bias", dt_float32, shape_t{(size_t)output_channels()});
    add_output("output", dt_float32,
               shape_t{input_shape[0], (size_t)output_channels(),
                       (size_t)get_kpu_pool_output_size((int32_t)input_shape[2],
                                                        pool_type_),
                       (size_t)get_kpu_pool_output_size((int32_t)input_shape[3],
                                                        pool_type_)})
        .attributes(cnctr_attr_no_layout_strides);
}

bool fake_kpu_conv2d::properties_equal(node &other) const {
    auto &r = static_cast<fake_kpu_conv2d &>(other);
    return is_depthwise() == r.is_depthwise() &&
           filter_type() == r.filter_type() && pool_type() == r.pool_type() &&
           fused_activation() == r.fused_activation();
}
