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

#define NTT_DEFINE_NATIVE_VECTOR(element_type, native_type, ...)               \
    namespace nncase::ntt::detail {                                            \
    template <>                                                                \
    class tensor_storage<element_type, fixed_shape<__VA_ARGS__>,               \
                         default_strides_t<fixed_shape<__VA_ARGS__>>, false,   \
                         true> {                                               \
        using shape_type = fixed_shape<__VA_ARGS__>;                           \
        using strides_type = default_strides_t<fixed_shape<__VA_ARGS__>>;      \
                                                                               \
      public:                                                                  \
        using buffer_type = native_type;                                       \
                                                                               \
        tensor_storage() = default;                                            \
        tensor_storage(buffer_type value) : buffer_(value) {}                  \
                                                                               \
        static constexpr auto shape() noexcept { return shape_type{}; }        \
        static constexpr auto strides() noexcept { return strides_type{}; }    \
        static constexpr size_t size() noexcept {                              \
            return linear_size(shape_type{}, strides_type{});                  \
        }                                                                      \
                                                                               \
        const buffer_type &buffer() const noexcept { return buffer_; }         \
        buffer_type &buffer() noexcept { return buffer_; }                     \
                                                                               \
        auto elements() const noexcept {                                       \
            return std::span<const element_type>(                              \
                reinterpret_cast<const element_type *>(&buffer_), size());     \
        }                                                                      \
        auto elements() noexcept {                                             \
            return std::span<element_type>(                                    \
                reinterpret_cast<element_type *>(&buffer_), size());           \
        }                                                                      \
                                                                               \
      private:                                                                 \
        buffer_type buffer_;                                                   \
    };                                                                         \
    }

namespace nncase::ntt {
template <class T, size_t... Lanes>
class vector : public tensor<T, fixed_shape<Lanes...>> {
  public:
    using tensor_type = tensor<T, fixed_shape<Lanes...>>;
    using tensor_type::tensor_type;

    static vector from_scalar(T v) noexcept;
};
} // namespace nncase::ntt
