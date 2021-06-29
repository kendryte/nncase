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

namespace nncase
{
inline std::vector<uint8_t> read_stream(std::istream &stream)
{
    stream.seekg(0, std::ios::end);
    size_t length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(length);
    stream.read(reinterpret_cast<char *>(data.data()), length);
    return data;
}

inline std::vector<uint8_t> read_file(const std::filesystem::path &filename)
{
    std::ifstream infile(filename.string(), std::ios::binary | std::ios::in);
    if (!infile.good())
        throw std::runtime_error("Cannot open file: " + filename.string());
    return read_stream(infile);
}
}
