/* Copyright 2020 Canaan Inc.
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
#include <filesystem>
#include <nncase/runtime/cpu/compiler_defs.h>

#ifdef WIN32
#include <Windows.h>
#endif

BEGIN_NS_NNCASE_RT_MODULE(cpu)

enum class shared_memory_openmode { create, open };

class shared_memory {
  public:
    shared_memory(const std::filesystem::path &path, size_t size,
                  shared_memory_openmode mode);
    ~shared_memory();

    gsl::byte *data() noexcept { return data_; };
    size_t size() const noexcept { return size_; }

  private:
#ifdef WIN32
    HANDLE handle_;
#else
    int fd_;
    std::string path_;
#endif
    gsl::byte *data_;
    size_t size_;
};

END_NS_NNCASE_RT_MODULE
