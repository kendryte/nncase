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
#include "shared_memory.h"
#include <iostream>

#ifdef WIN32
#include <Windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h> // for O_* constants
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // for mode constants
#include <unistd.h> // unlink
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

shared_memory::shared_memory(const std::filesystem::path &path, size_t size, shared_memory_openmode mode)
    :
#ifdef WIN32
    handle_(nullptr)
#else
    fd_(-1)
    , path_(path.c_str())
#endif
    , data_(nullptr)
    , size_(size)
{
    if (mode == shared_memory_openmode::create)
    {
#ifdef WIN32
        handle_ = CreateFileMappingW(nullptr, nullptr, PAGE_READWRITE, 0, (DWORD)size, path.c_str());
        if (!handle_)
            throw std::runtime_error("Failed to create shared memory.");
#else
        // shm segments persist across runs, and macOS will refuse
        // to ftruncate an existing shm segment, so to be on the safe
        // side, we unlink it beforehand.
        // TODO(amos) check errno while ignoring ENOENT?
        int ret = shm_unlink(path.c_str());
        if (ret < 0)
        {
            if (errno != ENOENT)
                throw std::runtime_error("Failed to unlink shared memory when create : " + std::string(std::strerror(errno)));
        }

        fd_ = shm_open(path.c_str(), O_CREAT | O_RDWR, 0755);
        if (fd_ < 0)
            throw std::runtime_error("Failed to open shared memory when create : " + std::string(std::strerror(errno)));

        // this is the only way to specify the size of a
        // newly-created POSIX shared memory object
        ret = ftruncate(fd_, size);
        if (ret != 0)
            throw std::runtime_error("Failed to ftruncate shared memory when create : " + std::string(std::strerror(errno)) + " " + std::to_string(size_) + " " + std::to_string(fd_));
#endif
    }
    else if (mode == shared_memory_openmode::open)
    {
#ifdef WIN32
        handle_ = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, path.c_str());
        if (!handle_)
            throw std::runtime_error("Failed to open shared memory.");
#else
        fd_ = shm_open(path.c_str(), O_RDWR, 0755);
        if (fd_ < 0)
            throw std::runtime_error("Failed to open shared memory.");
#endif
    }
    else
    {
        throw std::runtime_error("Invalid shared memory openmode.");
    }

#ifdef WIN32
    data_ = (gsl::byte *)MapViewOfFile(handle_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, size);
#else
    auto memory = mmap(nullptr, // addr
        size, // length
        PROT_READ | PROT_WRITE, // prot
        MAP_SHARED, // flags
        fd_, // fd
        0 // offset
    );

    if (memory == MAP_FAILED)
        throw std::runtime_error("Failed to map shared memory.");
    data_ = (gsl::byte *)memory;
#endif
}

shared_memory ::~shared_memory()
{
    if (data_)
    {
#ifdef WIN32
        UnmapViewOfFile(data_);
#else
        munmap(data_, size_);
#endif
    }

#ifdef WIN32
    CloseHandle(handle_);
    handle_ = nullptr;
#else
    close(fd_);
    shm_unlink(path_.c_str());
#endif
}
