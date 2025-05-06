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
#include <nncase/runtime/copy.h>
#include <cstring>

using namespace nncase;
using namespace nncase::runtime;

result<void> nncase::runtime::copy(
    host_buffer_t src_buffer, host_buffer_t dest_buffer, size_t src_start,
    size_t dest_start, datatype_t datatype, std::span<const size_t> shape,
    std::span<const size_t> src_strides,
    std::span<const size_t> dest_strides) noexcept { // Check for empty tensor
    if (shape.empty())
        return ok();

    const auto element_size = datatype->size_bytes();
    try_var(src_map, src_buffer->map(map_read));
    try_var(dest_map, dest_buffer->map(map_write));
    auto src = src_map.buffer().data() + src_start * element_size;
    auto dest = dest_map.buffer().data() + dest_start * element_size;

    // Calculate total number of elements
    int64_t num_elements = 1;
    for (const auto &dim : shape) {
        num_elements *= dim;
    }

    // Get the number of dimensions
    int ndim = shape.size();

    // Check if both tensors are contiguous in memory
    bool is_src_contiguous = true;
    bool is_dest_contiguous = true;

    int64_t expected_src_stride = element_size;
    int64_t expected_dest_stride = element_size;

    // Check strides from innermost dimension outward
    for (int i = ndim - 1; i >= 0; i--) {
        if (src_strides[i] != expected_src_stride) {
            is_src_contiguous = false;
        }
        if (dest_strides[i] != expected_dest_stride) {
            is_dest_contiguous = false;
        }
        expected_src_stride *= shape[i];
        expected_dest_stride *= shape[i];
    }

    // Fast path: if both tensors are contiguous, use memcpy
    if (is_src_contiguous && is_dest_contiguous) {
        std::memcpy(dest, src, num_elements * element_size);
        return ok();
    }

    // For 1D tensors or small tensors, use simple element-by-element copy
    if (ndim == 1 || num_elements < 1000) {
        std::vector<int64_t> indices(ndim, 0);

        // Simple loop for element-by-element copy
        while (true) {
            // Calculate offsets based on indices and strides
            int64_t src_offset = 0;
            int64_t dest_offset = 0;

            for (int i = 0; i < ndim; i++) {
                src_offset += indices[i] * src_strides[i];
                dest_offset += indices[i] * dest_strides[i];
            }

            // Copy single element
            std::memcpy(dest + dest_offset, src + src_offset, element_size);

            // Update indices (starting from innermost dimension)
            int dim = ndim - 1;
            while (dim >= 0) {
                indices[dim]++;
                if (indices[dim] < shape[dim]) {
                    break;
                }
                indices[dim] = 0;
                dim--;
            }

            // Exit when all elements have been processed
            if (dim < 0)
                break;
        }
        return ok();
    }

    // For multi-dimensional tensors, optimize inner loop
    std::vector<int64_t> indices(ndim, 0);

    while (true) {
        // Calculate base offsets for all dimensions except the innermost
        int64_t src_base_offset = 0;
        int64_t dest_base_offset = 0;

        for (int i = 0; i < ndim - 1; i++) {
            src_base_offset += indices[i] * src_strides[i];
            dest_base_offset += indices[i] * dest_strides[i];
        }

        // Check if innermost dimension is contiguous in both tensors
        if (src_strides[ndim - 1] == element_size &&
            dest_strides[ndim - 1] == element_size) {
            // Innermost dimension is contiguous, copy entire inner slice at
            // once
            size_t copy_size = shape[ndim - 1] * element_size;
            std::memcpy(dest + dest_base_offset, src + src_base_offset,
                        copy_size);
        } else {
            // Innermost dimension is not contiguous, copy elements one by one
            for (int64_t i = 0; i < shape[ndim - 1]; i++) {
                int64_t src_offset =
                    src_base_offset + i * src_strides[ndim - 1];
                int64_t dest_offset =
                    dest_base_offset + i * dest_strides[ndim - 1];
                std::memcpy(dest + dest_offset, src + src_offset, element_size);
            }
        }

        // Update indices for outer dimensions (all except innermost)
        int dim = ndim - 2;
        while (dim >= 0) {
            indices[dim]++;
            if (indices[dim] < shape[dim]) {
                break;
            }
            indices[dim] = 0;
            dim--;
        }

        // Exit when all elements have been processed
        if (dim < 0)
            break;
    }

    return ok();
}
