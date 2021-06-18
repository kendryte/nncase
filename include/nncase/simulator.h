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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nncase/runtime/datatypes.h>
#include <span>
#include <vector>

namespace nncase
{
struct simulate_options
{
    std::filesystem::path output_path;
    std::filesystem::path dataset;
    std::string dataset_format;
    std::function<void(size_t cnt, size_t total)> progress;

    std::string input_layout = "NCHW";
    float input_mean = 0.f;
    float input_std = 1.f;
};

class NNCASE_API simulator
{
public:
    static std::unique_ptr<simulator> create(std::vector<uint8_t> model, const simulate_options &options);

    virtual ~simulator();
    virtual void run() = 0;
};
}
