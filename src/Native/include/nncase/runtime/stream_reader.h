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
#include <gsl/gsl-lite.hpp>
#include <istream>
#include <iterator>
#include <nncase/compiler_defs.h>
#include <nncase/runtime/dbg.h>
#include <string>
#include <vector>

BEGIN_NS_NNCASE_RUNTIME

class stream_reader {
  public:
    stream_reader(std::istream &stream) : stream_(stream) {}

    std::streampos tell() const noexcept { return stream_.tellg(); }
    bool empty() const noexcept { return !stream_.eof(); }

    void seek(std::streampos pos) noexcept { stream_.seekg(pos); }

    template <class T> T read() {
        T value;
        read(value);
        return value;
    }

    template <class T> void read(T &value) {
        stream_.read(reinterpret_cast<char *>(&value), sizeof(value));
    }

    template <class T> void read_span(gsl::span<const T> &span, size_t size) {
        span = {reinterpret_cast<const T *>(begin_), size};
        advance(sizeof(T) * size);
    }

    template <class T = gsl::byte> gsl::span<const T> read_span(size_t size) {
        gsl::span<const T> span(reinterpret_cast<const T *>(begin_), size);
        advance(sizeof(T) * size);
        return span;
    }

    void read_avail(gsl::span<const gsl::byte> &span) {
        span = {begin_, end_};
        begin_ = end_;
    }

    gsl::span<const gsl::byte> read_until(gsl::byte value) {
        auto it = std::find(begin_, end_, value);
        return read_span((size_t)std::distance(begin_, it));
    }

    gsl::span<const gsl::byte> read_avail() {
        gsl::span<const gsl::byte> span;
        read_avail(span);
        return span;
    }

    gsl::span<const gsl::byte> peek_avail() { return {begin_, end_}; }

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

    void skip(size_t count) { stream_.seekg(count, std::ios::cur); }

  private:
    std::istream &stream_;
};

END_NS_NNCASE_RUNTIME
