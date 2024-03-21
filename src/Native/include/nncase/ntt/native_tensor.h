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
#include "tensor.h"

#define NTT_DEFINE_NATIVE_TENSOR(element_type, native_type, max_size)          \
    namespace nncase::ntt::detail {                                            \
    template <> class tensor_storage<element_type, max_size, false> {          \
      public:                                                                  \
        using buffer_type = native_type;                                       \
                                                                               \
        tensor_storage() = default;                                            \
        tensor_storage(std::in_place_t, buffer_type value) : buffer_(value) {} \
                                                                               \
        const buffer_type &buffer() const noexcept { return buffer_; }         \
        buffer_type &buffer() noexcept { return buffer_; }                     \
                                                                               \
        auto elements() const noexcept {                                       \
            return std::span<const element_type, max_size>(                    \
                reinterpret_cast<const element_type *>(&buffer_), max_size);   \
        }                                                                      \
        auto elements() noexcept {                                             \
            return std::span<element_type, max_size>(                          \
                reinterpret_cast<element_type *>(&buffer_), max_size);         \
        }                                                                      \
                                                                               \
      private:                                                                 \
        buffer_type buffer_;                                                   \
    };                                                                         \
    }
