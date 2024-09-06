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
#include <filesystem>
#include <fstream>
#include <vector>

namespace nncase {
template <typename T, std::size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;
    static constexpr std::size_t alignment = Alignment;

    aligned_allocator() noexcept = default;
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        
        if (auto ptr = static_cast<T*>(std::aligned_alloc(alignment, n * sizeof(T))))
            return ptr;

        throw std::bad_alloc();
    }

    void deallocate(T *p, std::size_t) noexcept { std::free(p); }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return true;
}

template <class T, class U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return false;
}


inline std::vector<uint8_t, aligned_allocator<uint8_t, 32>> read_stream(std::istream &stream) {
    stream.seekg(0, std::ios::end);
    size_t length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t, aligned_allocator<uint8_t, 32>> data(length);
    stream.read(reinterpret_cast<char *>(data.data()), length);
    return data;
}

inline std::vector<uint8_t, aligned_allocator<uint8_t, 32>> read_file(const std::filesystem::path &filename) {
    std::ifstream infile(filename.string(), std::ios::binary | std::ios::in);
    if (!infile.good())
        throw std::runtime_error("Cannot open file: " + filename.string());
    return read_stream(infile);
}
} // namespace nncase
