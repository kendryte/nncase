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
#include <nncase/ir/ops/k210/kpu_data_exchange.h>
#include <nncase/runtime/k210/runtime_module.h>
#include <nncase/runtime/k210/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;

kpu_upload::kpu_upload(shape_t input_shape) {
    module_type(k210_module_type);
    add_input("input", dt_uint8, input_shape)
        .attributes(cnctr_attr_no_layout_strides);
    add_output("output", dt_uint8, input_shape, mem_kpu)
        .attributes(cnctr_attr_no_layout_strides);
}

kpu_download::kpu_download(shape_t input_shape) {
    module_type(k210_module_type);
    add_input("input", dt_uint8, input_shape)
        .attributes(cnctr_attr_no_layout_strides);
    add_output("output", dt_uint8, input_shape)
        .attributes(cnctr_attr_no_layout_strides);
}
