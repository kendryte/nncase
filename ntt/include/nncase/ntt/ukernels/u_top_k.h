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
#include "../apply.h"
#include "../dimension.h"
#include "../primitive_ops.h"
#include "nncase/ntt/tensor_traits.h"
namespace nncase::ntt {
namespace ukernels {

template <bool Arch, Scalar TProbs, Scalar TIndices, size_t Rank, size_t Axis,
          bool Norm>
struct u_top_k {
  public:
    inline void operator()(int64_t inner_size, const TProbs *slice_input_ptr,
                           TProbs *slice_probs_ptr, TIndices *slice_indices_ptr,
                           int64_t input_stride, int64_t out_probs_stride,
                           int64_t out_indices_stride, int K, int64_t largest,
                           int64_t sorted) const {
        constexpr int64_t MAX_K = 128;

        // If K is small, use plug-in maintenance (register-friendly, reduce
        // index operations)
        if (K <= 8) {
            if (input_stride == 1) {
                small_k_path_stride1(inner_size, slice_input_ptr,
                                     slice_probs_ptr, slice_indices_ptr,
                                     out_probs_stride, out_indices_stride, K,
                                     largest, sorted);
            } else {
                small_k_path_strideN(inner_size, slice_input_ptr,
                                     slice_probs_ptr, slice_indices_ptr,
                                     input_stride, out_probs_stride,
                                     out_indices_stride, K, largest, sorted);
            }
            if constexpr (Norm) {
                normalize_probs(slice_probs_ptr, K, out_probs_stride);
            }
            return;
        }

        // Big K: Use handwritten heap (min-heap for largest, max-heap for
        // smallest)
        TProbs heap_vals[MAX_K];
        TIndices heap_ids[MAX_K];
        int heap_size = 0;

        if (largest) {
            // min-heap: heap_vals[0] is current smallest among top-K
            if (input_stride == 1) {
                // fast path: contiguous input
                const TProbs *p = slice_input_ptr;
                for (int64_t i = 0; i < inner_size; ++i, ++p) {
                    TProbs v = *p;
                    if (heap_size < K) {
                        heap_vals[heap_size] = v;
                        heap_ids[heap_size] = static_cast<TIndices>(i);
                        heapify_up_min(heap_vals, heap_ids, heap_size);
                        ++heap_size;
                    } else if (v > heap_vals[0]) {
                        heap_vals[0] = v;
                        heap_ids[0] = static_cast<TIndices>(i);
                        heapify_down_min(heap_vals, heap_ids, heap_size);
                    }
                }
            } else {
                for (int64_t i = 0; i < inner_size; ++i) {
                    TProbs v = slice_input_ptr[i * input_stride];
                    if (heap_size < K) {
                        heap_vals[heap_size] = v;
                        heap_ids[heap_size] = static_cast<TIndices>(i);
                        heapify_up_min(heap_vals, heap_ids, heap_size);
                        ++heap_size;
                    } else if (v > heap_vals[0]) {
                        heap_vals[0] = v;
                        heap_ids[0] = static_cast<TIndices>(i);
                        heapify_down_min(heap_vals, heap_ids, heap_size);
                    }
                }
            }

            if (sorted)
                selection_sort_desc(heap_vals, heap_ids, heap_size);

            // Write back (usually K is small, element-wise write overhead is
            // low)
            for (int i = 0; i < heap_size; ++i) {
                slice_probs_ptr[i * out_probs_stride] = heap_vals[i];
                slice_indices_ptr[i * out_indices_stride] = heap_ids[i];
            }
        } else {
            // smallest K: maintain max-heap (heap_vals[0] is largest among
            // bottom-K)
            if (input_stride == 1) {
                const TProbs *p = slice_input_ptr;
                for (int64_t i = 0; i < inner_size; ++i, ++p) {
                    TProbs v = *p;
                    if (heap_size < K) {
                        heap_vals[heap_size] = v;
                        heap_ids[heap_size] = static_cast<TIndices>(i);
                        heapify_up_max(heap_vals, heap_ids, heap_size);
                        ++heap_size;
                    } else if (v < heap_vals[0]) {
                        heap_vals[0] = v;
                        heap_ids[0] = static_cast<TIndices>(i);
                        heapify_down_max(heap_vals, heap_ids, heap_size);
                    }
                }
            } else {
                for (int64_t i = 0; i < inner_size; ++i) {
                    TProbs v = slice_input_ptr[i * input_stride];
                    if (heap_size < K) {
                        heap_vals[heap_size] = v;
                        heap_ids[heap_size] = static_cast<TIndices>(i);
                        heapify_up_max(heap_vals, heap_ids, heap_size);
                        ++heap_size;
                    } else if (v < heap_vals[0]) {
                        heap_vals[0] = v;
                        heap_ids[0] = static_cast<TIndices>(i);
                        heapify_down_max(heap_vals, heap_ids, heap_size);
                    }
                }
            }

            if (sorted)
                selection_sort_asc(heap_vals, heap_ids, heap_size);

            for (int i = 0; i < heap_size; ++i) {
                slice_probs_ptr[i * out_probs_stride] = heap_vals[i];
                slice_indices_ptr[i * out_indices_stride] = heap_ids[i];
            }
        }
        if constexpr (Norm) {
            normalize_probs(slice_probs_ptr, K, out_probs_stride);
        }
    }

  private:
    //  helper: small-K paths (K <= 8), keep ordered array (descending for
    //  largest))
    inline void small_k_path_stride1(int64_t inner_size, const TProbs *input,
                                     TProbs *out_probs, TIndices *out_indices,
                                     int64_t out_probs_stride,
                                     int64_t out_indices_stride, int K,
                                     int64_t largest,
                                     [[maybe_unused]] int64_t sorted) const {
        // Use a simple fixed-length array to keep selected values ​​in
        // order（largest: descending; smallest: ascending）
        TProbs topv[8];
        TIndices topi[8];
        int cur = 0; // Currently filled in number

        if (largest) {
            // keep topv[0] >= topv[1] >= ... >= topv[cur-1]
            for (int64_t i = 0; i < inner_size; ++i) {
                TProbs v = input[i];
                if (cur < K) {

                    // Insert into appropriate position and discard the smallest
                    int pos = cur++;
                    while (pos > 0 && topv[pos - 1] < v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                } else if (v > topv[cur - 1]) {
                    // Insert into appropriate position and discard minimum
                    int pos = cur - 1;
                    while (pos > 0 && topv[pos - 1] < v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                }
            }
            // It is already in descending order. If sorted requires output in
            // descending order, write it back directly.
            for (int i = 0; i < cur; ++i) {
                out_probs[i * out_probs_stride] = topv[i];
                out_indices[i * out_indices_stride] = topi[i];
            }
        } else {
            // smallest: keep ascending order topv[0] <= topv[1] <= ...
            for (int64_t i = 0; i < inner_size; ++i) {
                TProbs v = input[i];
                if (cur < K) {
                    int pos = cur++;
                    while (pos > 0 && topv[pos - 1] > v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                } else if (v < topv[cur - 1]) {
                    int pos = cur - 1;
                    while (pos > 0 && topv[pos - 1] > v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                }
            }
            for (int i = 0; i < cur; ++i) {
                out_probs[i * out_probs_stride] = topv[i];
                out_indices[i * out_indices_stride] = topi[i];
            }
        }
    }

    inline void small_k_path_strideN(int64_t inner_size, const TProbs *input,
                                     TProbs *out_probs, TIndices *out_indices,
                                     int64_t input_stride,
                                     int64_t out_probs_stride,
                                     int64_t out_indices_stride, int K,
                                     int64_t largest,
                                     [[maybe_unused]] int64_t sorted) const {
        TProbs topv[8];
        TIndices topi[8];
        int cur = 0;
        for (int64_t i = 0; i < inner_size; ++i) {
            TProbs v = input[i * input_stride];
            if (largest) {
                if (cur < K) {
                    int pos = cur++;
                    while (pos > 0 && topv[pos - 1] < v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                } else if (v > topv[cur - 1]) {
                    int pos = cur - 1;
                    while (pos > 0 && topv[pos - 1] < v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                }
            } else {
                if (cur < K) {
                    int pos = cur++;
                    while (pos > 0 && topv[pos - 1] > v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                } else if (v < topv[cur - 1]) {
                    int pos = cur - 1;
                    while (pos > 0 && topv[pos - 1] > v) {
                        topv[pos] = topv[pos - 1];
                        topi[pos] = topi[pos - 1];
                        --pos;
                    }
                    topv[pos] = v;
                    topi[pos] = static_cast<TIndices>(i);
                }
            }
        }
        for (int i = 0; i < cur; ++i) {
            out_probs[i * out_probs_stride] = topv[i];
            out_indices[i * out_indices_stride] = topi[i];
        }
    }

    // ---------------- heap helpers (min-heap)
    inline void heapify_up_min(TProbs *vals, TIndices *ids, int idx) const {
        while (idx > 0) {
            int parent = (idx - 1) >> 1;
            if (!(vals[idx] < vals[parent]))
                break;
            swap_node(vals, ids, idx, parent);
            idx = parent;
        }
    }
    inline void heapify_down_min(TProbs *vals, TIndices *ids,
                                 int heap_size) const {
        int idx = 0;
        while (true) {
            int left = (idx << 1) + 1;
            int right = left + 1;
            int smallest = idx;
            if (left < heap_size && vals[left] < vals[smallest])
                smallest = left;
            if (right < heap_size && vals[right] < vals[smallest])
                smallest = right;
            if (smallest == idx)
                break;
            swap_node(vals, ids, idx, smallest);
            idx = smallest;
        }
    }

    // ---------------- heap helpers (max-heap)
    inline void heapify_up_max(TProbs *vals, TIndices *ids, int idx) const {
        while (idx > 0) {
            int parent = (idx - 1) >> 1;
            if (!(vals[idx] > vals[parent]))
                break;
            swap_node(vals, ids, idx, parent);
            idx = parent;
        }
    }
    inline void heapify_down_max(TProbs *vals, TIndices *ids,
                                 int heap_size) const {
        int idx = 0;
        while (true) {
            int left = (idx << 1) + 1;
            int right = left + 1;
            int largest_i = idx;
            if (left < heap_size && vals[left] > vals[largest_i])
                largest_i = left;
            if (right < heap_size && vals[right] > vals[largest_i])
                largest_i = right;
            if (largest_i == idx)
                break;
            swap_node(vals, ids, idx, largest_i);
            idx = largest_i;
        }
    }

    // ---------------- selection sorts for final ordering
    inline void selection_sort_desc(TProbs *vals, TIndices *ids, int n) const {
        for (int i = 0; i < n - 1; ++i) {
            int max_i = i;
            for (int j = i + 1; j < n; ++j) {
                if (vals[j] > vals[max_i])
                    max_i = j;
            }
            if (max_i != i)
                swap_node(vals, ids, i, max_i);
        }
    }
    inline void selection_sort_asc(TProbs *vals, TIndices *ids, int n) const {
        for (int i = 0; i < n - 1; ++i) {
            int min_i = i;
            for (int j = i + 1; j < n; ++j) {
                if (vals[j] < vals[min_i])
                    min_i = j;
            }
            if (min_i != i)
                swap_node(vals, ids, i, min_i);
        }
    }

    inline void swap_node(TProbs *vals, TIndices *ids, int a, int b) const {
        TProbs tv = vals[a];
        vals[a] = vals[b];
        vals[b] = tv;
        TIndices ti = ids[a];
        ids[a] = ids[b];
        ids[b] = ti;
    }

    inline void normalize_probs(TProbs *slice_probs_ptr, int K,
                                int out_probs_stride) const {
        TProbs sum_probs = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum_probs += slice_probs_ptr[i * out_probs_stride];
        }

        for (int i = 0; i < K; ++i) {
            slice_probs_ptr[i * out_probs_stride] /= sum_probs;
        }
    }
};

} // namespace ukernels

template <Scalar TProbs, Scalar TIndices, size_t Rank, size_t Axis, bool Norm>
void u_top_k(int64_t inner_size, const TProbs *slice_input_ptr,
             TProbs *slice_probs_ptr, TIndices *slice_indices_ptr,
             int64_t input_stride, int64_t out_probs_stride,
             int64_t out_indices_stride, int K, int64_t largest,
             int64_t sorted) {

    ukernels::u_top_k<true, TProbs, TIndices, Rank, Axis, Norm> impl;
    impl(inner_size, slice_input_ptr, slice_probs_ptr, slice_indices_ptr,
         input_stride, out_probs_stride, out_indices_stride, K, largest,
         sorted);
}

} // namespace nncase::ntt