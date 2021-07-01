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
#include "compiler_defs.h"
#include <cstring>
#include <gsl/gsl-lite.hpp>

BEGIN_NS_NNCASE_RUNTIME

class span_reader
{
public:
    span_reader(gsl::span<const gsl::byte> span)
        : span_(span)
    {
    }

    bool empty() const noexcept { return span_.empty(); }
    size_t avail() const noexcept { return span_.size_bytes(); }

    template <class T>
    T read()
    {
        auto value = *reinterpret_cast<const T *>(span_.data());
        advance(sizeof(T));
        return value;
    }

    template <class T>
    T read_unaligned()
    {
        alignas(T) uint8_t storage[sizeof(T)];
        std::memcpy(storage, span_.data(), sizeof(T));
        advance(sizeof(T));
        return *reinterpret_cast<const T *>(storage);
    }

    template <class T>
    void read(T &value)
    {
        value = *reinterpret_cast<const T *>(span_.data());
        advance(sizeof(T));
    }

    template <class T>
    void read_span(gsl::span<const T> &span, size_t size)
    {
        span = { reinterpret_cast<const T *>(span_.data()), size };
        advance(sizeof(T) * size);
    }

    template <class T = gsl::byte>
    gsl::span<const T> read_span(size_t size)
    {
        gsl::span<const T> span(reinterpret_cast<const T *>(span_.data()), size);
        advance(sizeof(T) * size);
        return span;
    }

    void read_avail(gsl::span<const gsl::byte> &span)
    {
        span = span_;
        span_ = {};
    }

    gsl::span<const gsl::byte> read_avail()
    {
        auto span = span_;
        span_ = {};
        return span;
    }

    gsl::span<const gsl::byte> peek_avail()
    {
        return span_;
    }

    template <class T>
    T peek()
    {
        auto value = *reinterpret_cast<const T *>(span_.data());
        return value;
    }

    template <class T>
    T peek_unaligned()
    {
        T value;
        std::memcpy(&value, span_.data(), sizeof(T));
        return value;
    }

    template <class T>
    T peek_unaligned_with_offset(size_t offset)
    {
        T value;
        std::memcpy(&value, span_.data() + offset, sizeof(T));
        return value;
    }

    template <class T>
    const T *get_ref()
    {
        auto ptr = reinterpret_cast<const T *>(span_.data());
        advance(sizeof(T));
        return ptr;
    }

    template <class T>
    void get_ref(const T *&ptr)
    {
        ptr = get_ref<T>();
    }

    void skip(size_t count)
    {
        advance(count);
    }

private:
    void advance(size_t count)
    {
        span_ = span_.subspan(count);
    }

private:
    gsl::span<const gsl::byte> span_;
};

END_NS_NNCASE_RUNTIME
