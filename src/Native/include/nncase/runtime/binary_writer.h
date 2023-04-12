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
#include <cassert>
#include <iostream>
#include <span>

namespace nncase {
class binary_writer {
  public:
    binary_writer(std::ostream &stream)
        : stream_(stream), relative_offset_(0) {}

    template <class T> void write(T &&value) {
        stream_.write(reinterpret_cast<const char *>(&value), sizeof(value));
        relative_offset_ += sizeof(value);
    }

    template <class T> void write_array(std::span<T const> value) {
        stream_.write(reinterpret_cast<const char *>(value.data()),
                      value.size_bytes());
        relative_offset_ += value.size_bytes();
    }

    std::streampos position() const {
        assert(stream_);
        return stream_.tellp();
    }

    void position(std::streampos pos) {
        auto old_pos = position();
        stream_.seekp(pos);
        assert(stream_);
        relative_offset_ += pos - old_pos;
    }

    void skip(size_t len) {
        char zero = 0;
        for (size_t i = 0; i < len; i++)
            stream_.write(&zero, 1);
        relative_offset_ += len;
    }

    std::streamoff align_position(size_t alignment) {
        auto pos = position();
        auto rem = pos % alignment;
        if (rem != 0) {
            auto off = std::streamoff(alignment - rem);
            skip(off);
            return off;
        }

        return 0;
    }

    int64_t relative_offset() const noexcept { return relative_offset_; }

  private:
    std::ostream &stream_;
    int64_t relative_offset_;
};
} // namespace nncase
