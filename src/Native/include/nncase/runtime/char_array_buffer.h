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
        : data_(data), current_(data.data()) {}

  private:
    int_type underflow() {
        if (current_ == data_.end()) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_); // HERE!
    }

    int_type uflow() {
        if (current_ == data_.end()) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*current_++); // HERE!
    }

    int_type pbackfail(int_type ch) {
        if (current_ == data_.begin() ||
            (ch != traits_type::eof() && ch != current_[-1])) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*--current_); // HERE!
    }

    std::streamsize showmanyc() { return data_.size(); }

    gsl::span<const char> data_;
    const char *current_;
};
} // namespace nncase
