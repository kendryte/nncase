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
#include <limits>
#include <map>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <queue>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <typename T>
int64_t quick_partition(std::vector<std::pair<T, size_t>> &nums, int64_t lo, int64_t hi, bool largest)
{
    int64_t i = lo;
    int64_t j = hi + 1;
    T pivot = nums[lo].first;

    while (true)
    {
        if (largest)
        {
            while (++i < hi && nums[i].first > pivot)
                ;
            while (--j > lo && nums[j].first < pivot)
                ;
        }
        else
        {
            while (++i < hi && nums[i].first < pivot)
                ;
            while (--j > lo && nums[j].first > pivot)
                ;
        }

        if (i >= j)
        {
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
void quick_select(std::vector<std::pair<T, size_t>> &nums, int64_t lo, int64_t hi, int64_t k, bool largest)
{
    if (lo >= hi)
    {
        return;
    }

    int64_t idx = quick_partition(nums, lo, hi, largest);
    if (idx == k)
    {
        return;
    }

    return idx > k ? quick_select(nums, lo, idx - 1, k, largest) : quick_select(nums, idx + 1, hi, k, largest);
}

}

template result<void> reference::topk<float>(const float *input, float *output_values, NNCASE_UNUSED int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    NNCASE_UNUSED const runtime_shape_t &output_values_shape, const runtime_shape_t &output_values_strides,
    NNCASE_UNUSED const runtime_shape_t &output_indices_shape, NNCASE_UNUSED const runtime_shape_t &output_indices_strides,
    const int64_t k, const int32_t axis, const bool largest, const bool sorted) noexcept;

template <typename T>
result<void> reference::topk(const T *input, T *output_values, NNCASE_UNUSED int64_t *output_indices,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    NNCASE_UNUSED const runtime_shape_t &output_values_shape, const runtime_shape_t &output_values_strides,
    NNCASE_UNUSED const runtime_shape_t &output_indices_shape, NNCASE_UNUSED const runtime_shape_t &output_indices_strides,
    const int64_t k, const int32_t axis, const bool largest, const bool sorted) noexcept
{
    std::map<size_t, std::vector<std::pair<T, size_t>>> map;

    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto in_idx = offset(in_strides, index);
        runtime_shape_t axes { static_cast<size_t>(axis) };
        auto out_idx = offset(output_values_strides, kernels::detail::get_reduced_offset(index, axes, true));
        map[out_idx].push_back(std::make_pair(input[in_idx], index[axis]));
        return ok();
    }));

    size_t off = 1;
    for (auto i = static_cast<size_t>(axis) + 1; i < in_shape.size(); i++)
    {
        off *= in_shape[i];
    }

    for (auto e : map)
    {
        std::vector<size_t> indices(k, e.first);
        for (auto i = 1; i < k; i++)
        {
            indices[i] = indices[i - 1] + off;
        }

        if (sorted)
        {
            std::reverse(indices.begin(), indices.end());
            if (largest)
            {
                std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>, std::greater<std::pair<T, size_t>>> pq;
                for (auto &p : e.second)
                {
                    if (pq.size() < k)
                    {
                        pq.push(p);
                    }
                    else if (p.first > pq.top().first)
                    {
                        pq.pop();
                        pq.push(p);
                    }
                }

                // set the topk into output
                size_t idx = 0;
                while (!pq.empty())
                {
                    output_values[indices[idx]] = pq.top().first;
                    output_indices[indices[idx++]] = pq.top().second;
                    pq.pop();
                }
            }
            else
            {
                std::priority_queue<std::pair<T, size_t>, std::vector<std::pair<T, size_t>>, std::less<std::pair<T, size_t>>> pq;
                for (auto &p : e.second)
                {
                    if (pq.size() < k)
                    {
                        pq.push(p);
                    }
                    else if (p.first < pq.top().first)
                    {
                        pq.pop();
                        pq.push(p);
                    }
                }

                // set the topk into output
                size_t idx = 0;
                while (!pq.empty())
                {
                    output_values[indices[idx]] = pq.top().first;
                    output_indices[indices[idx++]] = pq.top().second;
                    pq.pop();
                }
            }
        }
        else
        {
            // not sort with topk
            quick_select(e.second, 0, static_cast<int64_t>(in_shape[axis] - 1), k, largest);

            // sort with idx
            // std::sort(e.second.begin(), e.second.begin() + k, [](std::pair<T, size_t> &a, std::pair<T, size_t> &b) { return a.second < b.second; });

            for (auto idx = 0; idx < k; idx++)
            {
                output_values[indices[idx]] = e.second[idx].first;
                output_indices[indices[idx]] = e.second[idx].second;
            }
        }
    }

    return ok();
}
