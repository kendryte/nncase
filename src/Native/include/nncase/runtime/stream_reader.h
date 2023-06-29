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

    template <class T> T read_unaligned() { return read<T>(); }

    template <class T> T peek() {
        T value;
        auto pos = tell();
        read(value);
        seek(pos);
        return value;
    }

    template <class T> T peek_unaligned() { return peek<T>(); }

    template <class T> void read(T &value) {
        stream_.read(reinterpret_cast<char *>(&value), sizeof(value));
    }

    template <class T> void read_span(gsl::span<T> span) {
        size_t sub_data_size = 8388608;
        for (size_t pos = 0; pos < span.size_bytes();) {
            if (pos + sub_data_size >= span.size_bytes())
                sub_data_size = span.size_bytes() - pos;
            stream_.read(reinterpret_cast<char *>(span.data()) + pos,
                         sub_data_size);
            pos += sub_data_size;
        }
    }

    void skip(size_t count) { stream_.seekg(count, std::ios::cur); }

  private:
    std::istream &stream_;
};

END_NS_NNCASE_RUNTIME
