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
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/util.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

template <typename T> static void copy_data(T *dst, const T *src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

#ifdef __riscv_vector
static void copy_data_rvv_i64(int64_t *dst, const int64_t *src, int n) {
    int i = n;
    while (i > 0) {
        size_t vl = vsetvl_e64m8(i);
        vint64m8_t v = vle64_v_i64m8(src, vl);
        vse64_v_i64m8(dst, v, vl);
        src += vl;
        dst += vl;
        i -= vl;
    }
}

static void broadcast_data_rvv_i64(int64_t *dst, int64_t val, int n) {
    while (n > 0) {
        size_t vl = vsetvl_e64m8(n);
        // 使用 vmv.v.x 将标量 val 广播到整个 vector 寄存器组
        vint64m8_t v = vmv_v_x_i64m8(val, vl);
        vse64_v_i64m8(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}
#endif

template <typename T>
result<void> tile_apply_impl(const T *input, T *output,
                             gsl::span<const size_t> in_shape,
                             gsl::span<const size_t> out_shape,
                             gsl::span<const size_t> in_strides,
                             gsl::span<const size_t> out_strides,
                             [[maybe_unused]] gsl::span<const size_t> repeats) {
    return apply(out_shape, [&](const auto &out_index) -> result<void> {
        auto in_index = dims_t(out_index.size());
        for (size_t i = 0; i < in_shape.size(); ++i) {
            in_index[i] = out_index[i] % in_shape[i];
        }
        output[offset(out_strides, out_index)] =
            input[offset(in_strides, in_index)];
        return ok();
    });
}

#ifdef __riscv_vector
template <typename T>
result<void> tile_impl(const T *input, T *output,
                       gsl::span<const size_t> in_shape,
                       gsl::span<const size_t> out_shape,
                       [[maybe_unused]] gsl::span<const size_t> in_strides,
                       [[maybe_unused]] gsl::span<const size_t> out_strides,
                       [[maybe_unused]] gsl::span<const size_t> &repeats);

template <>
result<void>
tile_impl<int64_t>(const int64_t *input, int64_t *output,
                   gsl::span<const size_t> in_shape,
                   gsl::span<const size_t> out_shape,
                   [[maybe_unused]] gsl::span<const size_t> in_strides,
                   [[maybe_unused]] gsl::span<const size_t> out_strides,
                   [[maybe_unused]] gsl::span<const size_t> &repeats) {
    size_t shape_size_in[4] = {1, 1, 1, 1};
    size_t shape_size_out[4] = {1, 1, 1, 1};
    size_t repeat_size[4] = {1, 1, 1, 1};
    for (int i = in_shape.size() - 1, j = 0; i >= 0; --i, ++j) {
        shape_size_in[j] = in_shape[i];
        shape_size_out[j] = out_shape[i];
    }
    for (int i = repeats.size() - 1, j = 0; i >= 0; --i, ++j) {
        repeat_size[j] = repeats[i];
    }

    auto w = shape_size_in[0];
    auto h = shape_size_in[1];
    auto d = shape_size_in[2];
    auto c = shape_size_in[3];

    auto wd = shape_size_out[0];
    auto hd = shape_size_out[1];
    auto dd = shape_size_out[2];

    auto repeat_w = repeat_size[0];
    auto repeat_h = repeat_size[1];
    auto repeat_d = repeat_size[2];
    auto repeat_c = repeat_size[3];

    bool is_w_broadcast = (w == 1);
    for (int ci = 0; ci < c; ++ci) {
        for (int di = 0; di < d; ++di) {
            for (int hi = 0; hi < h; ++hi) {
                const int64_t *src =
                    input + ci * h * d * w + di * h * w + hi * w;
                int64_t *dst =
                    output + ci * hd * dd * wd + di * hd * wd + hi * wd;

                if (is_w_broadcast) {
                    broadcast_data_rvv_i64(dst, *src, repeat_w);
                } else {
                    for (int i = 0; i < repeat_w; ++i) {
                        copy_data_rvv_i64(dst, src, w);
                        dst += w;
                    }
                }
            }
        }

        for (int di = 0; di < d; ++di) {
            int64_t *base_dst = output + ci * hd * dd * wd + di * hd * wd;
            auto block_size = h * wd;
            int64_t *dst1 = base_dst + block_size;

            for (int i = 1; i < repeat_h; ++i) {
                copy_data_rvv_i64(dst1, base_dst, block_size);
                dst1 += block_size;
            }
        }

        {
            int64_t *base_dst = output + ci * hd * dd * wd;
            auto block_size = d * hd * wd;
            int64_t *dst1 = base_dst + block_size;
            for (int i = 1; i < repeat_d; ++i) {
                copy_data_rvv_i64(dst1, base_dst, block_size);
                dst1 += block_size;
            }
        }
    }

    {
        int64_t *base_dst = output;
        auto block_size = c * dd * hd * wd;
        int64_t *dst1 = base_dst + block_size;
        for (int i = 1; i < repeat_c; ++i) {
            copy_data_rvv_i64(dst1, base_dst, block_size);
            dst1 += block_size;
        }
    }

    return ok();
}
#endif

template <typename T>
result<void> tile_impl(const T *input, T *output,
                       gsl::span<const size_t> in_shape,
                       gsl::span<const size_t> out_shape,
                       [[maybe_unused]] gsl::span<const size_t> in_strides,
                       [[maybe_unused]] gsl::span<const size_t> out_strides,
                       [[maybe_unused]] gsl::span<const size_t> &repeats) {
    size_t shape_size_in[4] = {1, 1, 1, 1};
    size_t shape_size_out[4] = {1, 1, 1, 1};
    size_t repeat_size[4] = {1, 1, 1, 1};
    for (int i = in_shape.size() - 1, j = 0; i >= 0; --i, ++j) {
        shape_size_in[j] = in_shape[i];
        shape_size_out[j] = out_shape[i];
    }

    for (int i = repeats.size() - 1, j = 0; i >= 0; --i, ++j) {
        repeat_size[j] = repeats[i];
    }

    auto w = shape_size_in[0];
    auto h = shape_size_in[1];
    auto d = shape_size_in[2];
    auto c = shape_size_in[3];

    auto wd = shape_size_out[0];
    auto hd = shape_size_out[1];
    auto dd = shape_size_out[2];

    auto repeat_w = repeat_size[0];
    auto repeat_h = repeat_size[1];
    auto repeat_d = repeat_size[2];
    auto repeat_c = repeat_size[3];

    for (int ci = 0; ci < c; ++ci) {
        for (int di = 0; di < d; ++di) {
            for (int hi = 0; hi < h; ++hi) {
                const T *src = input + ci * h * d * w + di * h * w + hi * w;
                T *dst = output + ci * hd * dd * wd + di * hd * wd + hi * wd;
                for (int i = 0; i < repeat_w; ++i) {
                    copy_data(dst, src, w);
                    dst += w;
                }
            }
        }
        for (int di = 0; di < d; ++di) {
            T *dst = output + ci * hd * dd * wd + di * hd * wd;
            auto size_x = h * wd;
            T *dst1 = dst + size_x;
            for (int i = 1; i < repeat_h; ++i) {
                copy_data(dst1, dst, size_x);
                dst1 += size_x;
            }
        }
        {
            T *dst = output + ci * hd * dd * wd;
            auto size_x = d * hd * wd;
            T *dst1 = dst + size_x;
            for (int i = 1; i < repeat_d; ++i) {
                copy_data(dst1, dst, size_x);
                dst1 += size_x;
            }
        }
    }

    {
        T *dst = output;
        auto size_x = c * dd * hd * wd;
        T *dst1 = dst + size_x;
        for (int i = 1; i < repeat_c; ++i) {
            copy_data(dst1, dst, size_x);
            dst1 += size_x;
        }
    }

    return ok();
}

#define TILE_IMPL(_ty)                                                         \
    return tile_impl(IN_CAST(_ty, input), OUT_CAST(_ty, output), in_shape,     \
                     out_shape, in_strides, out_strides, repeats);

result<void> nncase::kernels::stackvm::optimized::tile(
    datatype_t dt, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    gsl::span<const size_t> repeats) {
    if (in_shape.size() > 4) {
        return tile_apply_impl(input, output, in_shape, out_shape, in_strides,
                               out_strides, repeats);
    }
    try_var(tycode, to_typecode(dt));
    TYPE_SELECT(tycode, TILE_IMPL);
}