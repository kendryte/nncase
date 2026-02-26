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
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#if __riscv_vector
#include "utils.h"
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
template <class T, class IndicesT>
result<void> gather_elements_impl(
    const T *input, T *output,
    [[maybe_unused]] gsl::span<const size_t> in_shape,
    gsl::span<const size_t> out_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, const IndicesT *indices,
    gsl::span<const size_t> indices_shape, size_t axis,
    NNCASE_UNUSED kernel_context &context) noexcept {
    auto domain = out_shape.subspan(0, axis + 1);
    auto sub_in_strides = in_strides.subspan(0, axis + 1);
    auto sub_out_strides = out_strides.subspan(0, axis + 1);
    auto sub_indices_strides =
        gsl::span<const size_t>(get_default_strides(indices_shape))
            .subspan(0, axis + 1);

    size_t vl;
    return apply(
        domain, [&](gsl::span<const size_t> out_index) -> result<void> {
            dims_t in_index(out_index);
            in_index[axis] = 0;
            auto ptr_in = input + offset(sub_in_strides, in_index);
            auto ptr_idx = indices + offset(sub_indices_strides, out_index);
            auto ptr_out = output + offset(sub_out_strides, out_index);

            for (size_t j = 0; j < out_shape[axis + 1]; j += vl) {
                vl = vsetvl_e32m2(out_shape[axis + 1] - j);

                __rvv_int64m4_t v_idx_64 = vle64_v_i64m4(ptr_idx + j, vl);
                __rvv_int32m2_t v_idx = vncvt_x_x_w_i32m2(v_idx_64, vl);

                __rvv_int32m2_t v_offset =
                    vmul_vx_i32m2(v_idx, in_strides[axis], vl);

                __rvv_uint32m2_t v_col_id = vid_v_u32m2(vl);
                v_col_id = vadd_vx_u32m2(v_col_id, j, vl);

                v_offset = vadd_vv_i32m2(
                    v_offset, vreinterpret_v_u32m2_i32m2(v_col_id), vl);

                v_offset = vsll_vx_i32m2(v_offset, 2, vl);

                __rvv_float32m2_t v_data = vluxei32_v_f32m2(
                    ptr_in, vreinterpret_v_i32m2_u32m2(v_offset), vl);

                vse32_v_f32m2(ptr_out + j, v_data, vl);
            }

            return ok();
        });
}
} // namespace

result<void> nncase::kernels::stackvm::optimized::gather_elements(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    datatype_t indices_type, const gsl::byte *indices,
    gsl::span<const size_t> indices_shape, size_t axis,
    kernel_context &context) noexcept {
#if __riscv_vector
    if (axis == in_shape.size() - 2 &&
        type.as<prim_type_t>().unwrap()->typecode() == typecode_t::dt_float32) {
        return gather_elements_impl(reinterpret_cast<const float *>(input),
                                    reinterpret_cast<float *>(output), in_shape,
                                    out_shape, in_strides, out_strides,
                                    reinterpret_cast<const int64_t *>(indices),
                                    indices_shape, axis, context);
    } else {
        return reference::gather_elements(
            type, input, output, in_shape, out_shape, in_strides, out_strides,
            indices_type, indices, indices_shape, axis, context);
    }
#else
    return reference::gather_elements(type, input, output, in_shape, out_shape,
                                      in_strides, out_strides, indices_type,
                                      indices, indices_shape, axis, context);
#endif
}
