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
#include <ncnn/mat.h>
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/kernels/stackvm/optimized/opt_ops.h>

BEGIN_NS_NNCASE_RT_MODULE(ncnn)

inline result<::ncnn::Mat> to_ncnn_mat(value_t value) {
    try_var(t, value.as<tensor>());
    ::ncnn::Mat mat;
    auto shape = t->shape();
    auto elemsize = runtime::get_bytes(t->dtype());
    switch (shape.size()) {
    case 1:
        mat.create((int)shape[0], elemsize);
        break;
    case 2:
        mat.create((int)shape[1], (int)shape[0], elemsize);
        break;
    case 3:
        mat.create((int)shape[2], (int)shape[1], (int)shape[0], elemsize);
        break;
    case 4:
        mat.create((int)shape[3], (int)shape[2], (int)shape[1], (int)shape[0],
                   elemsize);
        break;
    default:
        return err(std::errc::invalid_argument);
    }

    try_var(hb, t->buffer().as_host());
    try_var(map, hb.map(map_read));
    return kernels::stackvm::optimized::slice(
        datatype, src_map.buffer().data() + src_start * datatype->size_bytes(),
        dest_map.buffer().data() + dest_start * datatype->size_bytes(), shape,
        src_strides, dest_strides, begins, ends, strides,
        kernels::default_kernel_context());
}

END_NS_NNCASE_RT_MODULE
