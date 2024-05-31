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
#include "../shape.h"
#include <vector>

namespace nncase::ntt::detail {
template <class T, size_t MaxSize, bool IsView> class tensor_storage;

// fixed tensor
template <class T, size_t MaxSize> class tensor_storage<T, MaxSize, false> {
  public:
    using buffer_type = std::array<T, MaxSize>;

    tensor_storage() = default;

    // ignore size
    explicit tensor_storage(size_t) noexcept {}
    tensor_storage(std::in_place_t, buffer_type value) noexcept
        : buffer_(value) {}

    constexpr const buffer_type &buffer() const noexcept { return buffer_; }
    constexpr buffer_type &buffer() noexcept { return buffer_; }

    constexpr std::span<const T, MaxSize> elements() const noexcept {
        return buffer_;
    }
    constexpr std::span<T, MaxSize> elements() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
};

// fixed view
template <class T, size_t MaxSize> class tensor_storage<T, MaxSize, true> {
  public:
    using buffer_type = std::span<T, MaxSize>;

    tensor_storage(std::in_place_t, buffer_type value) : buffer_(value) {}

    constexpr const buffer_type &buffer() const noexcept { return buffer_; }
    constexpr buffer_type &buffer() noexcept { return buffer_; }

    constexpr std::span<const T, MaxSize> elements() const noexcept {
        return buffer_;
    }
    constexpr std::span<T, MaxSize> elements() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
};

// dynamic tensor
template <class T> class tensor_storage<T, std::dynamic_extent, false> {
  public:
    using buffer_type = std::vector<T>;

    explicit tensor_storage(size_t size) : buffer_(size) {}
    tensor_storage(std::in_place_t, buffer_type value) : buffer_(value) {}

    constexpr const buffer_type &buffer() const noexcept { return buffer_; }
    constexpr buffer_type &buffer() noexcept { return buffer_; }

    constexpr std::span<const T> elements() const noexcept {
        return {buffer_.data(), buffer_.size()};
    }
    constexpr std::span<T> elements() noexcept {
        return {buffer_.data(), buffer_.size()};
    }

  private:
    buffer_type buffer_;
};

// dynamic view
template <class T> class tensor_storage<T, std::dynamic_extent, true> {
  public:
    using const_buffer_type = std::span<const T>;
    using buffer_type = std::span<T>;

    tensor_storage(std::in_place_t, buffer_type value) : buffer_(value) {}

    constexpr const_buffer_type buffer() const noexcept { return buffer_; }
    constexpr buffer_type buffer() noexcept { return buffer_; }

    constexpr const_buffer_type elements() const noexcept { return buffer_; }
    constexpr buffer_type elements() noexcept { return buffer_; }

  private:
    buffer_type buffer_;
};
} // namespace nncase::ntt::detail
