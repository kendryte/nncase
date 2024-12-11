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
#include <nncase/compiler_defs.h>
#include <nncase/runtime/result.h>
#include <string>
#include <vector>

BEGIN_NS_NNCASE_RUNTIME

class NNCASE_API stream {
  public:
    virtual ~stream() = default;

    virtual result<std::streampos> tell() const noexcept = 0;
    virtual result<void> seek(std::streamoff offset,
                              std::ios::seekdir dir) noexcept = 0;
    virtual result<size_t> read(void *buffer, size_t bytes) noexcept = 0;
    virtual result<void> write(const void *buffer, size_t bytes) noexcept = 0;
};

class NNCASE_API std_istream : public stream {
  public:
    std_istream(std::istream &stream) : stream_(stream) {}

    result<std::streampos> tell() const noexcept override {
        return stream_.fail() ? err(std::errc::io_error) : ok(stream_.tellg());
    }

    result<void> seek(std::streamoff offset,
                      std::ios::seekdir dir) noexcept override {
        stream_.seekg(offset, dir);
        return stream_.fail() ? err(std::errc::io_error) : ok();
    }

    result<size_t> read(void *buffer, size_t bytes) noexcept override {
        auto size = stream_.readsome(reinterpret_cast<char *>(buffer), bytes);
        return stream_.fail() ? err(std::errc::io_error) : ok(size);
    }

    result<void> write([[maybe_unused]] const void *buffer,
                       [[maybe_unused]] size_t bytes) noexcept override {
        return err(std::errc::not_supported);
    }

  private:
    std::istream &stream_;
};

END_NS_NNCASE_RUNTIME
