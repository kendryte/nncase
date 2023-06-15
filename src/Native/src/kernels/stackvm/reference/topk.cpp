
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
#include "ref_ops.h"
#include <map>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <queue>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::reference;

namespace {
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

result<void> kernels::stackvm::reference::topk(
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