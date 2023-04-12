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
#include "datatypes.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <gsl/gsl-lite.hpp>

namespace nncase::runtime {
class bitreader {
  public:
    bitreader(gsl::span<const uint8_t> data)
        : data_(data), buffer_(0), avail_(0) {}

    void read(uint8_t *dest, size_t bits) {
        while (bits) {
            auto to_read = std::min(bits, size_t(8));
            *dest++ = read_bits_le8(to_read);
            bits -= to_read;
        }
    }

    template <class T, size_t Bits> T read() {
        T ret{};
        read(reinterpret_cast<uint8_t *>(&ret), Bits);
        return ret;
    }

  private:
    uint8_t read_bits_le8(size_t bits) {
        assert(bits <= 8);

        fill_buffer_le8(bits);
        uint8_t ret = buffer_ & ((size_t(1) << bits) - 1);
        buffer_ >>= bits;
        avail_ -= bits;
        return ret;
    }

    void fill_buffer_le8(size_t bits) {
        if (avail_ < bits) {
            auto max_read_bytes =
                std::min(data_.size() * 8, sizeof(buffer_) * 8 - avail_) / 8;
            assert(max_read_bytes != 0);

            uint64_t tmp = 0;
            std::memcpy(&tmp, data_.data(), max_read_bytes);
            data_ = data_.subspan(max_read_bytes);
            buffer_ = buffer_ | (tmp << avail_);
            avail_ += max_read_bytes * 8;
        }
    }

  private:
    gsl::span<const uint8_t> data_;
    uint64_t buffer_;
    size_t avail_;
};

class bitwriter {
  public:
    bitwriter(gsl::span<uint8_t> data, size_t bitoffset = 0)
        : data_(data), buffer_(0), avail_(sizeof(buffer_) * 8) {
        if (bitoffset) {
            data_ = data_.subspan(bitoffset / 8);
            bitoffset %= 8;
            buffer_ = data_.front() & ((size_t(1) << bitoffset) - 1);
            avail_ -= bitoffset;
        }
    }

    ~bitwriter() { flush(); }

    void write(const uint8_t *src, size_t bits) {
        while (bits) {
            auto to_write = std::min(bits, size_t(8));
            write_bits_le8(*src++, to_write);
            bits -= to_write;
        }
    }

    template <size_t Bits, class T> void write(T value) {
        write(reinterpret_cast<const uint8_t *>(&value), Bits);
    }

    void flush() {
        auto write_bytes = (buffer_written_bits() + 7) / 8;
        if (write_bytes) {
            assert(data_.size() >= write_bytes);

            std::memcpy(data_.data(), &buffer_, write_bytes);
            data_ = data_.subspan(write_bytes);
            buffer_ = 0;
            avail_ = sizeof(buffer_) * 8;
        }
    }

  private:
    void write_bits_le8(uint8_t value, size_t bits) {
        assert(bits <= 8);

        reserve_buffer_8();
        size_t new_value = value & ((size_t(1) << bits) - 1);
        buffer_ = buffer_ | (new_value << buffer_written_bits());
        avail_ -= bits;
    }

    void reserve_buffer_8() {
        if (avail_ < 8) {
            auto write_bytes = buffer_written_bits() / 8;
            assert(data_.size() >= write_bytes);

            std::memcpy(data_.data(), &buffer_, write_bytes);
            data_ = data_.subspan(write_bytes);
            if (write_bytes == sizeof(buffer_))
                buffer_ = 0;
            else
                buffer_ >>= write_bytes * 8;
            avail_ += write_bytes * 8;
        }
    }

    size_t buffer_written_bits() const noexcept {
        return sizeof(buffer_) * 8 - avail_;
    }

  private:
    gsl::span<uint8_t> data_;
    uint64_t buffer_;
    size_t avail_;
};
} // namespace nncase::runtime
