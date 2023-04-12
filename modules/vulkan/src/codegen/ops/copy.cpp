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
#include "../module_builder.h"
#include <nncase/ir/op_utils.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/vulkan/runtime_types.h>
#include <vulkan/vulkan.hpp>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::vulkan;
using namespace nncase::ir;
using namespace nncase::runtime;
using namespace nncase::runtime::vulkan;
using namespace nlohmann;

namespace {
struct copy_shape_strides {
    shape_t shape;
    shape_t strides_shape1;
    shape_t strides_shape2;
};

copy_shape_strides optimize_copy_strides(copy_shape_strides src) {
    assert(src.strides_shape1.size() == src.strides_shape1.size());

    while (src.shape.size() > 1) {
        if (src.shape.back() == src.strides_shape1.back() &&
            src.shape.back() == src.strides_shape2.back()) {
            auto value = src.shape.back();
            src.shape.pop_back();
            src.strides_shape1.pop_back();
            src.strides_shape2.pop_back();
            src.shape.back() *= value;
            src.strides_shape1.back() *= value;
            src.strides_shape2.back() *= value;
        }
    }

    return src;
}
} // namespace

void vulkan_module_builder::emit(copy &node) {
    auto &tw = text_writer();
    auto &input = allocation(node.input());
    auto &output = allocation(node.input());

    ldbufbarrier_op_t in_bop{};
    in_bop.src_access_mask = (uint32_t)vk::AccessFlagBits::eMemoryRead;
    in_bop.dest_access_mask = 0;
    in_bop.memory = input.runtime_type();
    tw.write(in_bop);

    ldbufbarrier_op_t out_bop{};
    out_bop.src_access_mask = 0;
    out_bop.dest_access_mask = (uint32_t)vk::AccessFlagBits::eMemoryWrite;
    out_bop.memory = output.runtime_type();
    tw.write(out_bop);

    barrier_op_t bop{};
    bop.src_stage = (uint32_t)vk::PipelineStageFlagBits::eTransfer;
    bop.dest_stage = (uint32_t)vk::PipelineStageFlagBits::eTransfer;
    bop.buffer_barriers = 2;
    tw.write(bop);

    ldbuf(input.runtime_type());
    ldbuf(output.runtime_type());

    uint32_t regions = 0;
    auto opt_shape = optimize_copy_strides(
        {input.shape, input.strides_shape, output.strides_shape});
    if (opt_shape.shape.size() == 1) {
        ldbufcopy_op_t lbc_op;
        lbc_op.src = 0;
        lbc_op.dest = 0;
        lbc_op.size = (uint32_t)ir::get_bytes(input.type, opt_shape.shape);
        regions++;
        tw.write(lbc_op);
    } else {
        auto slice_len =
            (uint32_t)ir::get_bytes(input.type) * opt_shape.shape.back();
        auto src_strides = to_strides(opt_shape.strides_shape1);
        auto dest_strides = to_strides(opt_shape.strides_shape2);
        auto idx_shape = opt_shape.shape;
        idx_shape.back() = 1;
        kernels::cpu::reference::apply(
            idx_shape,
            [&](const shape_t &idx) -> result<void> {
                ldbufcopy_op_t lbc_op;
                lbc_op.src = kernels::offset(src_strides, idx);
                lbc_op.dest = kernels::offset(dest_strides, idx);
                lbc_op.size = slice_len;
                regions++;
                tw.write(lbc_op);
                return ok();
            })
            .unwrap_or_throw();
    }

    copybuf_op_t cb_op;
    cb_op.regions = regions;
    text_writer().write(cb_op);
}
