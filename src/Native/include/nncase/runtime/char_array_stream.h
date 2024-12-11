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
#include <cassert>
#include <span>

namespace nncase::runtime {
class char_array_stream : public stream {
  public:
    char_array_stream(std::span<const char> data)
        : begin_(data.data()),
          end_(data.data() + data.size()),
          current_(data.data()) {}

    result<std::streampos> tell() const noexcept override {
        return ok((std::streampos)(current_ - begin_));
    }

    result<void> seek(std::streamoff offset,
                      std::ios::seekdir dir) noexcept override {
        if (dir == std::ios_base::beg) {
            current_ = begin_ + offset;
        } else if (dir == std::ios_base::cur) {
            current_ += offset;
        } else if (dir == std::ios_base::end) {
            current_ = end_ + offset;
        }

        if (current_ < begin_ || current_ > end_)
            return err(std::errc::invalid_argument);

        return ok();
    }

    result<size_t> read(void *buffer, size_t bytes) noexcept override {
        std::streamsize available =
            static_cast<std::streamsize>(end_ - current_);
        std::streamsize n = (bytes > available) ? available : bytes;
        if (n > 0) {
            memcpy(buffer, current_, static_cast<size_t>(n));
            current_ += n;
        }
        return ok(n);
    }

    result<void> write(const void *buffer, size_t bytes) noexcept override {
        return err(std::errc::not_supported);
    }

    const char *const begin_;
    const char *const end_;
    const char *current_;
};
} // namespace nncase::runtime
