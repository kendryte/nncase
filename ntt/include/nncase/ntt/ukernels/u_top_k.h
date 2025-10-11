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
namespace nncase::ntt {
namespace ukernels {

template <bool Arch, Scalar TProbs, Scalar TIndices, size_t Rank, size_t Axis>
struct u_top_k {
  public:
    constexpr void operator()(int64_t inner_size, const TProbs *slice_input_ptr,
                              TProbs *slice_probs_ptr,
                              TIndices *slice_indices_ptr, int64_t input_stride,
                              int64_t out_probs_stride,
                              int64_t out_indices_stride, int K,
                              int64_t largest, int64_t sorted) const {
        if (K <= 0 || inner_size <= 0)
            return;
        if (K > inner_size)
            K = static_cast<int>(inner_size);

        constexpr int MAX_K = 128;

        TProbs heap_vals[MAX_K];
        int64_t heap_ids[MAX_K];
        int heap_size = 0;

        auto heap_swap = [&](int a, int b) {
            TProbs tv = heap_vals[a];
            heap_vals[a] = heap_vals[b];
            heap_vals[b] = tv;
            int64_t ti = heap_ids[a];
            heap_ids[a] = heap_ids[b];
            heap_ids[b] = ti;
        };

        if (largest) {
            auto heapify_up = [&](int idx) {
                while (idx > 0) {
                    int parent = (idx - 1) / 2;
                    if (!(heap_vals[idx] < heap_vals[parent]))
                        break;
                    heap_swap(idx, parent);
                    idx = parent;
                }
            };
            auto heapify_down = [&](int idx) {
                while (true) {
                    int left = idx * 2 + 1;
                    int right = left + 1;
                    int smallest = idx;
                    if (left < heap_size &&
                        heap_vals[left] < heap_vals[smallest])
                        smallest = left;
                    if (right < heap_size &&
                        heap_vals[right] < heap_vals[smallest])
                        smallest = right;
                    if (smallest == idx)
                        break;
                    heap_swap(idx, smallest);
                    idx = smallest;
                }
            };

            for (int64_t i = 0; i < inner_size; ++i) {
                TProbs v = slice_input_ptr[i * input_stride];
                if (heap_size < K) {
                    heap_vals[heap_size] = v;
                    heap_ids[heap_size] = i;
                    heapify_up(heap_size);
                    ++heap_size;
                } else if (v > heap_vals[0]) {
                    heap_vals[0] = v;
                    heap_ids[0] = i;
                    heapify_down(0);
                }
            }

            if (sorted) {
                for (int i = 0; i < heap_size - 1; ++i) {
                    int max_i = i;
                    for (int j = i + 1; j < heap_size; ++j) {
                        if (heap_vals[j] > heap_vals[max_i])
                            max_i = j;
                    }
                    if (max_i != i)
                        heap_swap(i, max_i);
                }
            }

            for (int i = 0; i < heap_size; ++i) {
                slice_probs_ptr[i * out_probs_stride] = heap_vals[i];
                slice_indices_ptr[i * out_indices_stride] =
                    static_cast<TIndices>(heap_ids[i]);
            }
        } else {
            auto heapify_up = [&](int idx) {
                while (idx > 0) {
                    int parent = (idx - 1) / 2;
                    if (!(heap_vals[idx] > heap_vals[parent]))
                        break;
                    heap_swap(idx, parent);
                    idx = parent;
                }
            };
            auto heapify_down = [&](int idx) {
                while (true) {
                    int left = idx * 2 + 1;
                    int right = left + 1;
                    int largest_i = idx;
                    if (left < heap_size &&
                        heap_vals[left] > heap_vals[largest_i])
                        largest_i = left;
                    if (right < heap_size &&
                        heap_vals[right] > heap_vals[largest_i])
                        largest_i = right;
                    if (largest_i == idx)
                        break;
                    heap_swap(idx, largest_i);
                    idx = largest_i;
                }
            };

            for (int64_t i = 0; i < inner_size; ++i) {
                TProbs v = slice_input_ptr[i * input_stride];
                if (heap_size < K) {
                    heap_vals[heap_size] = v;
                    heap_ids[heap_size] = i;
                    heapify_up(heap_size);
                    ++heap_size;
                } else if (v < heap_vals[0]) {
                    heap_vals[0] = v;
                    heap_ids[0] = i;
                    heapify_down(0);
                }
            }

            if (sorted) {
                for (int i = 0; i < heap_size - 1; ++i) {
                    int min_i = i;
                    for (int j = i + 1; j < heap_size; ++j) {
                        if (heap_vals[j] < heap_vals[min_i])
                            min_i = j;
                    }
                    if (min_i != i)
                        heap_swap(i, min_i);
                }
            }

            for (int i = 0; i < heap_size; ++i) {
                slice_probs_ptr[i * out_probs_stride] = heap_vals[i];
                slice_indices_ptr[i * out_indices_stride] =
                    static_cast<TIndices>(heap_ids[i]);
            }
        }
    }
};

} // namespace ukernels

template <Scalar TProbs, Scalar TIndices, size_t Rank, size_t Axis>
void u_top_k(int64_t inner_size, const TProbs *slice_input_ptr,
             TProbs *slice_probs_ptr, TIndices *slice_indices_ptr,
             int64_t input_stride, int64_t out_probs_stride,
             int64_t out_indices_stride, int K, int64_t largest,
             int64_t sorted) {

    ukernels::u_top_k<true, TProbs, TIndices, Rank, Axis> impl;
    impl(inner_size, slice_input_ptr, slice_probs_ptr, slice_indices_ptr,
         input_stride, out_probs_stride, out_indices_stride, K, largest,
         sorted);
}

} // namespace nncase::ntt