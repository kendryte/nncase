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
#include "stream.h"
#include <cstring>
#include <nncase/compiler_defs.h>
#include <nncase/runtime/dbg.h>

BEGIN_NS_NNCASE_RUNTIME

class stream_reader {
  public:
    stream_reader(runtime::stream &stream) : stream_(stream) {}

    std::streampos tell() const noexcept { return stream_.tell().unwrap(); }
    void seek(std::streampos pos) noexcept {
        stream_.seek(pos, std::ios::beg).unwrap();
    }

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
        auto size =
            stream_.read(reinterpret_cast<char *>(&value), sizeof(value))
                .unwrap_or_throw();
        if (size != sizeof(value))
            std::abort();
    }

    template <class T> void read_span(std::span<T> span) {
        auto size =
            stream_
                .read(reinterpret_cast<char *>(span.data()), span.size_bytes())
                .unwrap_or_throw();
        if (size != span.size_bytes())
            std::abort();
    }

    void skip(size_t count) { stream_.seek(count, std::ios::cur).unwrap(); }

  private:
    runtime::stream &stream_;
};

END_NS_NNCASE_RUNTIME
