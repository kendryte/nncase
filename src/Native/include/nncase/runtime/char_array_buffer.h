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
#include <gsl/gsl-lite.hpp>
#include <iostream>

namespace nncase {
class char_array_buffer : public std::streambuf {
  public:
    char_array_buffer(gsl::span<const char> data)
        : begin_(data.begin()), end_(data.end()), current_(data.data()) {}

  private:
    int_type underflow() {
        if (current_ == end_)
            return traits_type::eof();

        return traits_type::to_int_type(*current_);
    }

    int_type uflow() {
        if (current_ == end_)
            return traits_type::eof();

        return traits_type::to_int_type(*current_++);
    }

    int_type pbackfail(int_type ch) {
        if (current_ == begin_ ||
            (ch != traits_type::eof() && ch != current_[-1]))
            return traits_type::eof();

        return traits_type::to_int_type(*--current_);
    }

    std::streamsize showmanyc() {
        assert(std::less_equal<const char *>()(current_, end_));
        return end_ - current_;
    }

    std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
                           [[maybe_unused]] std::ios_base::openmode which) {
        if (way == std::ios_base::beg) {
            current_ = begin_ + off;
        } else if (way == std::ios_base::cur) {
            current_ += off;
        } else if (way == std::ios_base::end) {
            current_ = end_ + off;
        }

        if (current_ < begin_ || current_ > end_)
            return -1;

        return current_ - begin_;
    }

    std::streampos seekpos(std::streampos sp,
                           [[maybe_unused]] std::ios_base::openmode which) {
        current_ = begin_ + sp;

        if (current_ < begin_ || current_ > end_)
            return -1;

        return current_ - begin_;
    }

    const char *const begin_;
    const char *const end_;
    const char *current_;
};
} // namespace nncase
