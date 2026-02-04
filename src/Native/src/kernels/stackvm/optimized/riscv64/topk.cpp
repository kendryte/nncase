
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
#include <map>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <queue>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
#ifdef __riscv_vector
template <typename T>
void update_queue(
    std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>,
                        std::greater<>> &pq,
    T val, size_t idx, T &threshold) {
    if (val > threshold) {
        pq.pop();
        pq.emplace(val, idx);
        threshold = pq.top().first; // 更新门槛，越来越高
    }
}

void topK_rvv_f32(const float *input, float *output, int64_t *indices,
                  size_t length, int64_t k) {

    std::priority_queue<std::pair<float, size_t>,
                        std::vector<std::pair<float, size_t>>, std::greater<>>
        pq;

    for (int i = 0; i < k; ++i) {
        pq.emplace(input[i], i);
    }

    float threshold = pq.top().first;

    size_t vl_max = vsetvlmax_e32m8();
    std::vector<float> temp_vals(vl_max);
    std::vector<uint32_t> temp_idxs(vl_max);

    for (size_t i = k, vl = 0; i < length; i += vl) {
        vl = vsetvl_e32m8(length - i);

        vfloat32m8_t v_data = vle32_v_f32m8(input + i, vl);

        vbool4_t mask = vmfgt_vf_f32m8_b4(v_data, threshold, vl);

        long count = vcpop_m_b4(mask, vl);

        if (count == 0) {
            continue;
        }

        vuint32m8_t v_seq = vid_v_u32m8(vl);
        vuint32m8_t v_indices = vadd_vx_u32m8(v_seq, i, vl);

        vfloat32m8_t v_active_vals;
        v_active_vals = vcompress_vm_f32m8(mask, v_active_vals, v_data, vl);

        vuint32m8_t v_active_idxs;
        v_active_idxs = vcompress_vm_u32m8(mask, v_active_idxs, v_indices, vl);

        vse32_v_f32m8(temp_vals.data(), v_active_vals, count);
        vse32_v_u32m8(temp_idxs.data(), v_active_idxs, count);

        for (int j = 0; j < count; ++j) {
            update_queue(pq, temp_vals[j], temp_idxs[j], threshold);
        }
    }

    for (int j = k - 1; j >= 0; j--) {
        auto top = pq.top();
        output[j] = top.first;
        indices[j] = top.second;
        pq.pop();
    }
}
#endif
template <typename T>
int64_t quick_partition(std::vector<std::pair<T, size_t>> &nums, int64_t lo,
                        int64_t hi, bool largest) {
    int64_t i = lo;
    int64_t j = hi + 1;
    T pivot = nums[lo].first;

    while (true) {
        if (largest) {
            while (++i < hi && nums[i].first > pivot)
                ;
            while (--j > lo && nums[j].first < pivot)
                ;
        } else {
            while (++i < hi && nums[i].first < pivot)
                ;
            while (--j > lo && nums[j].first > pivot)
                ;
        }

        if (i >= j) {
            break;
        }

        std::swap(nums[i].first, nums[j].first);
        std::swap(nums[i].second, nums[j].second);
    }

    std::swap(nums[lo].first, nums[j].first);
    std::swap(nums[lo].second, nums[j].second);

    return j;
}

template <typename T>
void quick_select(std::vector<std::pair<T, size_t>> &nums, int64_t lo,
                  int64_t hi, int64_t k, bool largest) {
    if (lo >= hi) {
        return;
    }

    int64_t idx = quick_partition(nums, lo, hi, largest);
    if (idx == k) {
        return;
    }

    return idx > k ? quick_select(nums, lo, idx - 1, k, largest)
                   : quick_select(nums, idx + 1, hi, k, largest);
}

} // namespace

template <typename T> struct Compare {
    const T *value;
    bool operator()(size_t a, size_t b) { return value[a] < value[b]; }
};

template <typename T>
void topK(const T *input, T *output, int64_t *indices, size_t length,
          int64_t k) {
    std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>,
                        std::greater<>>
        topK_index;
    for (int i = 0; i < length; i++) {
        if (topK_index.size() >= k && input[i] > topK_index.top().first) {
            topK_index.pop();
            topK_index.emplace(input[i], i);
        } else if (topK_index.size() < k) {
            topK_index.emplace(input[i], i);
        }
    }
    for (int j = k - 1; j >= 0; j--) {
        auto top = topK_index.top();
        output[j] = top.first;
        indices[j] = top.second;
        topK_index.pop();
    }
}

template <typename T>
result<void>
topk_impl(const T *input, T *output_values, int64_t *output_indices,
          gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
          gsl::span<const size_t> output_values_shape,
          gsl::span<const size_t> output_values_strides,
          gsl::span<const size_t> output_indices_shape,
          gsl::span<const size_t> output_indices_strides, const int64_t k,
          const int32_t axis, const bool largest, const bool sorted) noexcept {
    (void)output_values_shape;
    (void)output_indices_shape;
    (void)output_indices_strides;

    // naive implementation of default attributes
    if (sorted && largest && (axis == -1 || axis == in_shape.size() - 1)) {
        auto outer_loop_cnt = compute_size(in_shape) / in_shape.back();
        for (auto i = 0; i < outer_loop_cnt; ++i) {
            auto input_ptr = input + i * in_shape.back();
            int64_t *output_indices_ptr =
                output_indices + i * output_indices_shape.back();
            T *output_values_ptr =
                output_values + i * output_values_shape.back();
#ifdef __riscv_vector
            if (largest && std::is_same<T, float>::value) {
                topK_rvv_f32((const float *)input_ptr,
                             (float *)output_values_ptr, output_indices_ptr,
                             in_shape.back(), k);
            } else {
                topK(input_ptr, output_values_ptr, output_indices_ptr,
                     in_shape.back(), k);
            }
#else
            topK(input_ptr, output_values_ptr, output_indices_ptr,
                 in_shape.back(), k);
#endif
        }
        return ok();
    }

    std::map<size_t, std::vector<std::pair<T, size_t>>> map;

    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto in_idx = offset(in_strides, index);
        dims_t axes{static_cast<size_t>(axis)};
        auto out_idx =
            offset(output_values_strides,
                   kernels::detail::get_reduced_offset(index, axes, true));
        map[out_idx].push_back(std::make_pair(input[in_idx], index[axis]));
        return ok();
    }));

    size_t off = 1;
    for (auto i = static_cast<size_t>(axis) + 1; i < in_shape.size(); i++) {
        off *= in_shape[i];
    }

    for (auto e : map) {
        std::vector<size_t> indices(k, e.first);
        for (auto i = 1; i < k; i++) {
            indices[i] = indices[i - 1] + off;
        }

        if (sorted) {
            std::reverse(indices.begin(), indices.end());
            if (largest) {
                std::priority_queue<std::pair<T, size_t>,
                                    std::vector<std::pair<T, size_t>>,
                                    std::greater<std::pair<T, size_t>>>
                    pq;
                for (auto &p : e.second) {
                    if (pq.size() < static_cast<size_t>(k)) {
                        pq.push(p);
                    } else if (p.first > pq.top().first) {
                        pq.pop();
                        pq.push(p);
                    }
                }

                // set the topk into output
                size_t idx = 0;
                while (!pq.empty()) {
                    output_values[indices[idx]] = pq.top().first;
                    output_indices[indices[idx++]] = pq.top().second;
                    pq.pop();
                }
            } else {
                std::priority_queue<std::pair<T, size_t>,
                                    std::vector<std::pair<T, size_t>>,
                                    std::less<std::pair<T, size_t>>>
                    pq;
                for (auto &p : e.second) {
                    if (pq.size() < static_cast<size_t>(k)) {
                        pq.push(p);
                    } else if (p.first < pq.top().first) {
                        pq.pop();
                        pq.push(p);
                    }
                }

                // set the topk into output
                size_t idx = 0;
                while (!pq.empty()) {
                    output_values[indices[idx]] = pq.top().first;
                    output_indices[indices[idx++]] = pq.top().second;
                    pq.pop();
                }
            }
        } else {
            // not sort with topk
            quick_select(e.second, 0, static_cast<int64_t>(in_shape[axis] - 1),
                         k, largest);

            // sort with idx
            // std::sort(e.second.begin(), e.second.begin() + k, [](std::pair<T,
            // size_t> &a, std::pair<T, size_t> &b) { return a.second <
            // b.second; });

            for (auto idx = 0; idx < k; idx++) {
                output_values[indices[idx]] = e.second[idx].first;
                output_indices[indices[idx]] = e.second[idx].second;
            }
        }
    }

    return ok();
}

#define TOPK_IMPL(_ty)                                                         \
    return topk_impl(IN_CAST(_ty, input), OUT_CAST(_ty, output_values),        \
                     output_indices, in_shape, in_strides,                     \
                     output_values_shape, output_values_strides,               \
                     output_indices_shape, output_indices_strides, k, axis,    \
                     largest, sorted)

result<void> kernels::stackvm::optimized::topk(
    typecode_t typecode, const gsl::byte *input, gsl::byte *output_values,
    int64_t *output_indices, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides,
    gsl::span<const size_t> output_values_shape,
    gsl::span<const size_t> output_values_strides,
    gsl::span<const size_t> output_indices_shape,
    gsl::span<const size_t> output_indices_strides, const int64_t k,
    const int32_t axis, const bool largest, const bool sorted) noexcept {
    TYPE_SELECT(typecode, TOPK_IMPL);
}
