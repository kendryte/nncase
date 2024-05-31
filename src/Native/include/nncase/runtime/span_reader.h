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
#include <cstring>
#include <iterator>
#include <nncase/compiler_defs.h>
#include <nncase/runtime/dbg.h>
#include <span>
#include <string>
#include <vector>

BEGIN_NS_NNCASE_RUNTIME

class span_reader {
  public:
    span_reader(std::span<const std::byte> span)
        : begin_(span.data()), end_(span.data() + span.size()) {}

    const std::byte *tell() const noexcept { return begin_; }
    bool empty() const noexcept { return begin_ == end_; }
    size_t avail() const noexcept { return end_ - begin_; }

    void seek(const std::byte *pos) noexcept { begin_ = pos; }

    template <class T> T read() {
        auto value = *reinterpret_cast<const T *>(begin_);
        advance(sizeof(T));
        return value;
    }

    template <class T> T read_unaligned() {
        alignas(T) uint8_t storage[sizeof(T)];
        std::memcpy(storage, begin_, sizeof(T));
        advance(sizeof(T));
        return *reinterpret_cast<const T *>(storage);
    }

    template <class T> void read(T &value) {
        value = *reinterpret_cast<const T *>(begin_);
        advance(sizeof(T));
    }

    template <class T> void read_span(std::span<const T> &span, size_t size) {
        span = {reinterpret_cast<const T *>(begin_), size};
        advance(sizeof(T) * size);
    }

    template <class T> void read_span(std::span<T> span) {
        std::memcpy(span.data(), begin_, span.size_bytes());
        advance(span.size_bytes());
    }

    template <class T = std::byte> std::span<const T> read_span(size_t size) {
        std::span<const T> span(reinterpret_cast<const T *>(begin_), size);
        advance(sizeof(T) * size);
        return span;
    }

    std::string read_string() {
        auto span = read_until((std::byte)0);
        advance(1);
        return {reinterpret_cast<const char *>(span.data()), span.size()};
    }

    std::vector<std::string> read_string_array() {
        std::vector<std::string> array;
        while (true) {
            if (peek<char>() == '\0') {
                advance(1);
                break;
            }
            array.emplace_back(read_string());
        }
        return array;
    }

    void read_avail(std::span<const std::byte> &span) {
        span = {begin_, end_};
        begin_ = end_;
    }

    std::span<const std::byte> read_until(std::byte value) {
        auto it = std::find(begin_, end_, value);
        return read_span((size_t)std::distance(begin_, it));
    }

    std::span<const std::byte> read_avail() {
        std::span<const std::byte> span;
        read_avail(span);
        return span;
    }

    std::span<const std::byte> peek_avail() { return {begin_, end_}; }

    template <class T> T peek_with_offset(size_t offset) {
        auto value = *reinterpret_cast<const T *>(begin_ + offset);
        return value;
    }

    template <class T> T peek() { return peek_with_offset<T>(0); }

    template <class T> T peek_unaligned_with_offset(size_t offset) {
        T value;
        std::memcpy(&value, begin_ + offset, sizeof(T));
        return value;
    }

    template <class T> T peek_unaligned() {
        return peek_unaligned_with_offset<T>(0);
    }

    template <class T> const T *peek_ref() {
        auto ptr = reinterpret_cast<const T *>(begin_);
        return ptr;
    }

    template <class T> const T *get_ref() {
        auto ptr = peek_ref<T>();
        advance(sizeof(T));
        return ptr;
    }

    template <class T> void get_ref(const T *&ptr) { ptr = get_ref<T>(); }

    void skip(size_t count) { advance(count); }

  private:
    void advance(size_t count) {
        begin_ += count;
        dbg_check(begin_ <= end_);
    }

  private:
    const std::byte *begin_;
    const std::byte *end_;
};

END_NS_NNCASE_RUNTIME
